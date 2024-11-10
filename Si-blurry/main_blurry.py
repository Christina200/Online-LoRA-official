# import torch
import time
from configuration import config_blurry
from datasets import *
from methods.er_baseline import ER
from methods.mvp import MVP
from online_lora import Online_LoRA_Trainer

# torch.backends.cudnn.enabled = False
methods = { "er"    : ER, 
            "mvp"   : MVP
            }

def main():
    # Get Configurations
    args = config_blurry.base_parser()
    print(args)

    if args.mode == "olora": 
        trainer = Online_LoRA_Trainer(**vars(args))
    else: trainer = methods[args.mode](**vars(args))
    
    start_time = time.time()
    trainer.run()
    end_time = time.time()

    print(f"Runtime of the program is {(end_time - start_time)/60} minutes. ")

if __name__ == "__main__":
    main()