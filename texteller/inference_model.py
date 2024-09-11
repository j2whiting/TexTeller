import cv2 as cv
import numpy as np
from texteller.models.ocr_model.utils.to_katex import to_katex
from texteller.models.ocr_model.utils.inference import inference as latex_inference
from texteller.models.ocr_model.model.TexTeller import TexTeller


class InferenceModel:
    def __init__(self, inference_mode='cpu', num_beams=1, mix_mode=False):
        self.inference_mode = inference_mode
        self.num_beams = num_beams
        self.mix_mode = mix_mode

        print('Loading LaTeX recognition model and tokenizer...')
        self.latex_rec_model = TexTeller.from_pretrained()
        self.tokenizer = TexTeller.get_tokenizer()
        print('LaTeX recognition model and tokenizer loaded.')

    def predict(self, image_bytes):
        """
        Call like so:
        
        m = MixedInferenceModel()
        with open ('/Users/jwhiting/Documents/gollm_evaluation_ASKEM_may_2024/equation_from_pdf/formula_28.png', 'rb') as img:
	    img = img.read()
    	output = m.predict(img)
        assert isinstance(output, str)
        
        """
        image_nparray = np.frombuffer(image_bytes, np.uint8)
        img = cv.imdecode(image_nparray, cv.IMREAD_COLOR)

        print('Performing LaTeX recognition inference...')
        res = latex_inference(self.latex_rec_model, self.tokenizer, [img], self.inference_mode, self.num_beams)
        res = to_katex(res[0])

        return res
