## Florence-2
Apresentamos o Florence-2, um novo modelo de base de visão com uma representação unificada e baseada em prompt para uma variedade de tarefas de visão computacional e linguagem de visão. Embora os grandes modelos de visão existentes se destaquem na aprendizagem por transferência, eles lutam para executar uma diversidade de tarefas com instruções simples, uma capacidade que implica lidar com a complexidade de várias hierarquias espaciais e granularidade semântica. O Florence-2 foi projetado para receber prompts de texto como instruções de tarefa e gerar resultados desejáveis ​​em formas de texto, seja legenda, detecção de objetos, aterramento ou segmentação. Essa configuração de aprendizagem multitarefa exige dados anotados em larga escala e alta qualidade.

## Método

Florence-2 pode interpretar prompts de texto simples para executar tarefas como legendas, detecção de objetos e segmentação. Ele aproveita nosso conjunto de dados FLD-5B, contendo 5,4 bilhões de anotações em 126 milhões de imagens, para dominar o aprendizado multitarefa.

## Como utilizar!

Use o código abaixo para começar com o modelo. Todos os modelos são treinados com float16.
        
        import requests

        import torch
        from PIL import Image
        from transformers import AutoProcessor, AutoModelForCausalLM 


        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
        processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

        prompt = "<OD>"

        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
        image = Image.open(requests.get(url, stream=True).raw)

        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

        generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
        
        print(parsed_answer)


## Tarefas

Este modelo é capaz de executar diferentes tarefas por meio da alteração dos prompts.

Primeiro, vamos definir uma função para executar um prompt.

        import requests

        import torch
        from PIL import Image
        from transformers import AutoProcessor, AutoModelForCausalLM 

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
        processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
        image = Image.open(requests.get(url, stream=True).raw)

        def run_example(task_prompt, text_input=None):
        if text_input is None:
        prompt = task_prompt
        else:
        prompt = task_prompt + text_input
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
        generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))

        print(parsed_answer)

Aqui estão as tarefas Florence-2que você pode executar:

## Rubrica

        prompt = "<CAPTION>"
        run_example(prompt)

## Legenda detalhada

        prompt = "<DETAILED_CAPTION>"
        run_example(prompt)

## Legenda mais detalhada

        prompt = "<MORE_DETAILED_CAPTION>"
        run_example(prompt)

## Legenda da frase Grounding

A tarefa de fixação da legenda à frase requer entrada de texto adicional, ou seja, legenda.
Formato dos resultados de aterramento da legenda da frase: {'<CAPTION_TO_PHRASE_GROUNDING>': {'bboxes': [[x1, y1, x2, y2], ...], 'labels': ['', '', ...]}}

        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        results = run_example(task_prompt, text_input="A green car parked in front of a yellow building.")

## Detecção de objetos

        prompt = "<OD>"
        run_example(prompt)

## Legenda da região densa

        prompt = "<DENSE_REGION_CAPTION>"
        run_example(prompt)

## Proposta de região

        prompt = "<REGION_PROPOSAL>"
        run_example(prompt)

## Reconhecimento óptico de caracteres

        prompt = "<OCR>"
        run_example(prompt)

## OCR com região

        prompt = "<OCR_WITH_REGION>"
        run_example(prompt)

https://huggingface.co/microsoft/Florence-2-large
