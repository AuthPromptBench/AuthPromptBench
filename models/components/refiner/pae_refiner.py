from models.utils.PAE.configs import get_configs
import torch
from torch import nn
from models.utils.PAE.gpt import GPTActor
import os
from transformers import GPT2Tokenizer
from typing import List, Callable
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

DEFAULT_CFG = get_configs("gpt2-medium")


class PAE_refiner(nn.Module):
    def __init__(
            self, 
            configs_name: str = "gpt2-medium", 
            ckpt_path: str = os.path.join(PROJECT_ROOT, "checkpoints/PAE/actor_step3000.pt")):
        super(PAE_refiner, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = get_configs(configs_name)
        self.model = GPTActor.from_checkpoint(self.cfg, ckpt_path).to(self.device)
        self.model.eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        self.step_dict={
            0: torch.tensor(self.tokenizer.encode("0-0.5"),device=self.device),#0-0.5
            1: torch.tensor(self.tokenizer.encode("0-1"),device=self.device), #0-1
            2: torch.tensor(self.tokenizer.encode("0.5-1"),device=self.device),#0.5-1
        }
        self.w_dict={
            0: torch.tensor(self.tokenizer.encode("0.5"),device=self.device),
            1: torch.tensor(self.tokenizer.encode("0.75"),device=self.device), 
            2: torch.tensor(self.tokenizer.encode("1.0"),device=self.device),
            3: torch.tensor(self.tokenizer.encode("1.25"),device=self.device), 
            4: torch.tensor(self.tokenizer.encode("1.5"),device=self.device),
        }
        self.token_dict={
            ",": torch.tensor(self.tokenizer.encode(",")[0],device=self.device),
            ".": torch.tensor(self.tokenizer.encode(".")[0],device=self.device),
            ":": torch.tensor(self.tokenizer.encode(":")[0],device=self.device),
            " [": torch.tensor(self.tokenizer.encode(" [")[0],device=self.device),
            "[": torch.tensor(self.tokenizer.encode("[")[0],device=self.device),
            "]": torch.tensor(self.tokenizer.encode("]")[0],device=self.device),
            " ": torch.tensor(self.tokenizer.encode(" ")[0],device=self.device)
        }

        self.pattern = r'\[([^]]*):0-1:1\.0\]'#r'\[(\s*\w+):0-1:1\.0\]'

    

    def prepare_gpt2_input(self, prompt: str):
        enc = self.tokenizer
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
        indices = encode(prompt)
        x = (torch.tensor(indices, dtype=torch.long, device=self.device)[None, ...])
        return x, decode
    
    def trans_token(self, bef_list,diffw_list,diffstep_list):
        if len(bef_list)==0:
            return bef_list
        aft_list=torch.tensor([],device=self.device)

        ind=0
        token = bef_list[ind]
        if not (token==self.token_dict[","] or token==self.token_dict["."]): 
            special_token_ind_list=[]

            while not (token==self.token_dict[","] or token==self.token_dict[","] or token==self.token_dict[" "] or self.tokenizer.decode([token.long()]).startswith(" ")): 
                token = bef_list[ind]
                aft_list=torch.cat([aft_list,token.unsqueeze(0)])
                ind+=1
                
                if ind>=(len(bef_list)):
                    break
            if ind<(len(bef_list)):
                token = bef_list[ind]
            while ind<(len(bef_list)) and not (token==self.token_dict[","] or token==self.token_dict["."]): 
                if token==self.token_dict[" "] or token==self.token_dict[","] or token==self.token_dict["."]:
                    aft_list=torch.cat([aft_list,token.unsqueeze(0)])
                    ind+=1
                    if ind>=(len(bef_list)):
                        break
                    token = bef_list[ind]
                else:
                    special_token_ind_list.append(ind)
                    
                    ind+=1
                    if ind>=(len(bef_list)):
                        break
                    token = bef_list[ind]
                    

                    if token ==self.token_dict[","] or token==self.token_dict["."]:
                        break


            try:
                w_counts = torch.bincount(diffw_list[special_token_ind_list])
                w_mode=int(torch.argmax(w_counts).item())
            except:
                w_mode=2

            try:
                counts = torch.bincount(diffstep_list[special_token_ind_list])
                mode=int(torch.argmax(counts).item())
            except:
                mode=1
            

            for ind in special_token_ind_list:
                
                aft_list=torch.cat([aft_list,self.token_dict["["].unsqueeze(0)])
                s_token = bef_list[ind]
                
                aft_list=torch.cat([aft_list,s_token.unsqueeze(0)])
                aft_list=torch.cat([aft_list,self.token_dict[":"].unsqueeze(0)])
                aft_list=torch.cat([aft_list,self.step_dict[mode]])
                aft_list=torch.cat([aft_list,self.token_dict[":"].unsqueeze(0)])
                aft_list=torch.cat([aft_list,self.w_dict[w_mode]])
                aft_list=torch.cat([aft_list,self.token_dict["]"].unsqueeze(0)])
            ind+=1

            while ind < len(bef_list):
                
                token = bef_list[ind]    

                if not (token==self.token_dict[","] or token==self.token_dict["."]): 
                    aft_list=torch.cat([aft_list,token.unsqueeze(0)])
                    ind+=1
                else: 
                    ind+=1
                    if ind >= len(bef_list):
                        break
                    token = bef_list[ind]
                    special_token_ind_list=[]
                    while not (token==self.token_dict[","] or token==self.token_dict["."]): 

                        special_token_ind_list.append(ind)
                        
                        ind+=1
                        if ind>=(len(bef_list)):
                            break
                        token = bef_list[ind]
                        

                        if token ==self.token_dict[","] or token==self.token_dict["."]:
                            break


                    aft_list=torch.cat([aft_list,self.token_dict[","].unsqueeze(0)])
                    try:
                        w_counts = torch.bincount(diffw_list[special_token_ind_list])
                        w_mode=int(torch.argmax(w_counts).item())
                    except:
                        w_mode=2

                    try:
                        counts = torch.bincount(diffstep_list[special_token_ind_list])
                        mode=int(torch.argmax(counts).item())
                    except:
                        mode=1
                    

                    for ind in special_token_ind_list:
                        
                        aft_list=torch.cat([aft_list,self.token_dict["["].unsqueeze(0)])
                        s_token = bef_list[ind]
                        
                        aft_list=torch.cat([aft_list,s_token.unsqueeze(0)])
        
                        aft_list=torch.cat([aft_list,self.token_dict[":"].unsqueeze(0)])


                        aft_list=torch.cat([aft_list,self.step_dict[mode]])
        
                        aft_list=torch.cat([aft_list,self.token_dict[":"].unsqueeze(0)])
        
                        aft_list=torch.cat([aft_list,self.w_dict[w_mode]])
        
                        aft_list=torch.cat([aft_list,self.token_dict["]"].unsqueeze(0)])
                    ind+=1

        
        else:
            while ind < len(bef_list):
                
                token = bef_list[ind]    

                if not (token==self.token_dict[","] or token==self.token_dict["."]): 
                    aft_list=torch.cat([aft_list,token.unsqueeze(0)])
                    ind+=1
                else: 
                    ind+=1
                    if ind >= len(bef_list):
                        break
                    token = bef_list[ind]
                    special_token_ind_list=[]
                    while not (token==self.token_dict[","] or token==self.token_dict["."]): 

                        special_token_ind_list.append(ind)
                        
                        ind+=1
                        if ind>=(len(bef_list)):
                            break
                        token = bef_list[ind]
                        

                        if token ==self.token_dict[","] or token==self.token_dict["."]:
                            break


                    aft_list=torch.cat([aft_list,self.token_dict[","].unsqueeze(0)])
                    try:
                        w_counts = torch.bincount(diffw_list[special_token_ind_list])
                        w_mode=int(torch.argmax(w_counts).item())
                    except:
                        w_mode=2

                    try:
                        counts = torch.bincount(diffstep_list[special_token_ind_list])
                        mode=int(torch.argmax(counts).item())
                    except:
                        mode=1
                    

                    for ind in special_token_ind_list:
                        
                        aft_list=torch.cat([aft_list,self.token_dict["["].unsqueeze(0)])
                        s_token = bef_list[ind]
                        
                        aft_list=torch.cat([aft_list,s_token.unsqueeze(0)])
        
                        aft_list=torch.cat([aft_list,self.token_dict[":"].unsqueeze(0)])


                        aft_list=torch.cat([aft_list,self.step_dict[mode]])
        
                        aft_list=torch.cat([aft_list,self.token_dict[":"].unsqueeze(0)])
        
                        aft_list=torch.cat([aft_list,self.w_dict[w_mode]])
        
                        aft_list=torch.cat([aft_list,self.token_dict["]"].unsqueeze(0)])
                    ind+=1

            
        return aft_list

    @torch.inference_mode()
    def generate(self, prompt: str) -> str:
        temperature = 0.9
        top_k = 200

        x, decode = self.prepare_gpt2_input(prompt)
        max_new_tokens = 75-x.shape[-1]
        y, diffw_list, diffstep_list = self.model.generate_dy(x,
                            max_new_tokens,
                            temperature=temperature,
                            top_k=top_k)

        if y.shape==torch.Size([0]):
            return prompt
        y_0=y[0].long()

        input_w=diffw_list[0].long()
        input_step=diffstep_list[0].long()

        target_value = torch.tensor(50256,device=self.device)


        end = (y_0 == target_value).nonzero(as_tuple=True)[0]
        if end.numel() > 0:
            y_0 = y_0[:end[0]]
            input_w=input_w[:end[0]]
            input_step=input_step[:end[0]]

        res=decode(torch.cat([x[0],self.trans_token(y_0, input_w, input_step)]))
        end = res.find("[<|endoftext|>")
        if end > 0:
            res= res[:end]

        end = res.find("<|endoftext|>")
        if end > 0:
            res=res[:end]

        return res
    
    def generate_and_save(self, prompts: List[str], save_function: Callable):
        refined_texts = []
        for prompt in tqdm(prompts, desc="Refining prompts with PAE"):
            refined_text = self.generate(prompt)
            refined_texts.append(refined_text)
        save_function(refined_texts)
    


if __name__ == "__main__":
    pae_refiner = PAE_refiner(ckpt_path=os.path.join(PROJECT_ROOT, "checkpoints/PAE/sd14/actor_step3000.pt"))
    prompt = 'a photo of a happy cat'
    result = pae_refiner.generate(prompt)
    print(result)