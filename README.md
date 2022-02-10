# ErgoChairML

**ErgoChairML** aims to generate chair for a poseture. The chair image is generated using a pix2pix GAN trained on custom dataset. The dataset is created by scraping Google Image with key words and preprocessing the image with movenet and Mask RCNN.

The project is packaged as python package gen_chair. And Demo is available on Google Colab. [Media Processing](https://colab.research.google.com/github/Victoooooor/ErgoChairML/blob/main/Demo.ipynb) Demo is made to generate image from image or video uploads, and [Webcam Demo](https://colab.research.google.com/github/Victoooooor/ErgoChairML/blob/main/Demo_Vid.ipynb) generates images through webcam feed. 

# Files

	./examples						#code snippets
		Preprocess.py					#single person detection
		Preprocess_multi.py				#multi detecction
		TensorBoard.ipynb				#notebook for TensorBoard
		movenet_thunder.tflite			#movenet trained model from tfhub
		scraper.py						#Google Image scraper
		test.ipynb						#test trained pix2pix
		test.py							#test trained pix2pix
		train.py						#train pix2pix
	./gen_chair 					#package root
		./ml 							#movenet implementation
		./mrcnn							#Mask-RCNN Image Segmentation
		./tracker						#part of movenet
		gen_multi.py					#multi detection wrapper
		gen_pose.py						#sinple detection
		image_scraper.py				#Google Image Scraper
		pix2pix.py						#modified pix2pix
		segmentation.py					#Mask-RCNN wrapper
		
		

## Installation and Usage

	Install:
		pip install git+https://github.com/Victoooooor/ErgoChairML.git
	
	Use:
		import gen_chair
	

## Produced Work

	The following are outputs of the Demo.ipynb using trained model loaded on GCS.
	Each produced image is concatenated from 3 square images: 
		Left  	-> 	Input
		Middle	->	Output of Preprocessing
		Right	-> 	GAN Output
![Skeleton Output 2](https://github.com/Victoooooor/ErgoChairML/blob/main/image/ske1.jpg?raw=true)
![Skeleton Output 2](https://github.com/Victoooooor/ErgoChairML/blob/main/image/ske2.jpg?raw=true)
![Skeleton Output 3](https://github.com/Victoooooor/ErgoChairML/blob/main/image/ske3.jpg?raw=true)
![Skeleton Output 4](https://github.com/Victoooooor/ErgoChairML/blob/main/image/ske4.jpg?raw=true)
![Skeleton Animated 5](https://github.com/Victoooooor/ErgoChairML/blob/main/image/animated.gif?raw=true)
![Skeleton Output 6](https://github.com/Victoooooor/ErgoChairML/blob/main/image/weirdpose.jpg?raw=true)
![Webcam Output 1](https://github.com/Victoooooor/ErgoChairML/blob/main/image/webcam4.jpg?raw=true)
Design based off ErgoChair output by Rhyna
![enter image description here](https://github.com/Victoooooor/ErgoChairML/blob/main/image/work1.png?raw=true)
Work produced by added depth to ErgoChair output

![Work2
](https://github.com/Victoooooor/ErgoChairML/blob/main/image/work2.gif?raw=true)
![Work3
](https://github.com/Victoooooor/ErgoChairML/blob/main/image/work3.gif?raw=true)



## Diagrams

![Simple Diagram](https://github.com/Victoooooor/ErgoChairML/blob/main/image/diagram.png?raw=true)
