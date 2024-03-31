from llava.model.builder import load_pretrained_model


if __name__ == '__main__':
    model_name = 'llava-v1.6-qwen2moe'
    # model_path = '/home/thesby/projects/pretrained_models/LLM/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4'
    model_path = '/home/thesby/projects/pretrained_models/LLM/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4'

    model = load_pretrained_model(model_name=model_name, model_path=model_path,
                                  load_4bit=False, device='cuda', model_base=None)
    print(model)