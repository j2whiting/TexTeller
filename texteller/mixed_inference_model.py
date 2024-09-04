import os
from pathlib import Path
import cv2 as cv
import numpy as np
from onnxruntime import InferenceSession
from texteller.models.third_party.paddleocr.infer import predict_det, predict_rec
from texteller.models.third_party.paddleocr.infer import utility
from texteller.models.utils import mix_inference
from texteller.models.ocr_model.utils.to_katex import to_katex
from texteller.models.ocr_model.utils.inference import inference as latex_inference
from texteller.models.ocr_model.model.TexTeller import TexTeller
from texteller.models.det_model.inference import PredictConfig

class MixedInferenceModel:
    def __init__(self, inference_mode='cpu', num_beams=1, mix_mode=False):
        self.inference_mode = inference_mode
        self.num_beams = num_beams
        self.mix_mode = mix_mode

        print('Loading LaTeX recognition model and tokenizer...')
        self.latex_rec_model = TexTeller.from_pretrained()
        self.tokenizer = TexTeller.get_tokenizer()
        print('LaTeX recognition model and tokenizer loaded.')

        if self.mix_mode:
            print('Loading models and configurations for mixed inference...')
            self.infer_config = PredictConfig("./models/det_model/model/infer_cfg.yml")
            self.latex_det_model = InferenceSession("./models/det_model/model/rtdetr_r50vd_6x_coco.onnx")
            
            self.det_model_dir = "./models/third_party/paddleocr/checkpoints/det/default_model.onnx"
            self.rec_model_dir = "./models/third_party/paddleocr/checkpoints/rec/default_model.onnx"
            
            SIZE_LIMIT = 20 * 1024 * 1024
            self.use_gpu = inference_mode == 'cuda'
            self.det_use_gpu = False
            self.rec_use_gpu = self.use_gpu and not (os.path.getsize(self.rec_model_dir) < SIZE_LIMIT)

            paddleocr_args = utility.parse_args()
            paddleocr_args.use_onnx = True
            paddleocr_args.det_model_dir = self.det_model_dir
            paddleocr_args.rec_model_dir = self.rec_model_dir

            paddleocr_args.use_gpu = self.det_use_gpu
            self.detector = predict_det.TextDetector(paddleocr_args)
            paddleocr_args.use_gpu = self.rec_use_gpu
            self.recognizer = predict_rec.TextRecognizer(paddleocr_args)
            
            self.lang_ocr_models = [self.detector, self.recognizer]
            print('Models and configurations for mixed inference loaded.')

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

        if not self.mix_mode:
            print('Performing LaTeX recognition inference...')
            res = latex_inference(self.latex_rec_model, self.tokenizer, [img], self.inference_mode, self.num_beams)
            res = to_katex(res[0])
        else:
            print('Performing mixed inference...')
            res = mix_inference(
                img,
                self.infer_config,
                self.latex_det_model,
                self.lang_ocr_models,
                [self.latex_rec_model, self.tokenizer],
                self.inference_mode,
                self.num_beams
            )
        
        return res
