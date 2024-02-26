import google.generativeai as genai
import os
from tqdm import tqdm
import PIL.Image

class GeminiVisionApi():
    
    def __init__(self,API_KEY):
        self.API_KEY=API_KEY

    def input_to_ai(self,image_name,prompt):
        genai.configure(api_key=self.API_KEY)
        img = PIL.Image.open(image_name)
        model = genai.GenerativeModel('gemini-pro-vision')
        safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]
        response = model.generate_content([prompt, img], safety_settings=safety_settings,stream=True)
        response.resolve()
        return response.text.replace('*','').replace('\n','')

    
    
    def save_caption(self,text,output_dir,image_name,file_extension):
        f=open(output_dir+str(image_name).replace(file_extension,'.txt'),'w')
        f.write(text)
        f.close()

def main():
    your_api_key= 'Your API Key' 

    file_extension='.jpg' #what extension do image have

    input_dir='dir/' #image dir where image is
    output_dir='dir/' #where you want to store caption

    prompt='prompt' #your prompt

    gpt=GeminiVisionApi(your_api_key)
    already_done=[]
    
    for filename in (os.listdir(input_dir)):
        if '.txt' in filename: 
            imgname=filename.replace('.txt',file_extension)
            if file_extension in imgname:
                already_done.append(imgname)
    
    new_img_list = [item for item in os.listdir(input_dir) if item not in already_done]
    
    for filename in tqdm(new_img_list,desc="Processing items", unit="item"):
       
            if filename.endswith(file_extension):

                try:
                    result=gpt.input_to_ai(input_dir+filename,prompt)
                    gpt.save_caption(result,output_dir,filename,file_extension)
                    pass
                except Exception as e:
                    print(f"Error: {e}")

            
if __name__=='__main__':
    main()

