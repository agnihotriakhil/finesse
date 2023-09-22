
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os


from huggingface_hub import model_info
from train_text_to_image_lora import base_model_name


def generate_image(pipe, prompt):

    image = pipe(prompt, num_inference_steps=20).images[0]
    image_name = prompt[:min(10,len(prompt))] + '.jpg'
    save_path = os.path.join('./stable-diffusion/', image_name)
    image.save(save_path)

    return image_name


def main():

    pipe = StableDiffusionPipeline.from_pretrained(base_model_name)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    prompt = str(input('\n\n Please enter a prompt to generate an image:  '))
    image_name = generate_image(pipe, prompt)
    print('Image for this prompt is saved as ', image_name)

    do_again = str(input('\n\n Do you want to generate more images? [y/n]  : '))
    if(do_again == 'y'):
        main()
    elif(do_again == 'n'):
        print('\n\nok bye')
    else:
        print('Please enter y or n only')


if(__name__ == '__main__'):
    main()