## CNN Classifier for Just-in-Time Woodpeckers Detection and Deterrent

## Table of contents
- [Introduction](#introduction)
- [Publications](#publications)
- [Hardware](#hardware)
- [Software](#software)
- [Roadmap](#roadmap)
- [License](#license)
- [Project Status](#status)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## Introduction <a name="introduction"></a>
Woodpeckers can cause significant damage to homes, especially in suburban areas. There are a number of preventing and repelling methods including passive decoys, though these may only provide temporary relief. Subsequently, it may be more efficient to implement a woodpecker deterrent, such as motion, light, sound, or ultrasound that would be triggered by detection of woodpecker signature drumming. To detect the typical 25 Hz drumming frequency, sampling periods under 10 milliseconds with frequent FFTs are required with considerable computational costs. An in-hardware spectrum analyzer may avoid these costs by trading off frequency for time resolutions. The trained model converted to TF Lite Micro, ported to an MCU, and identifies a variety of the prerecorded woodpecker drumming. 

This methodology could be applied to similar problems where is required to identify relatively fast audio signals computing on edge devices.

## Publications <a name="publications"></a>
Alexander Greysukh, "CNN Classifier for Just-in-Time Woodpeckers Detection and Deterrent",  [https://arxiv.org/abs/2107.10676](https://arxiv.org/abs/2107.10676)

## Hardware <a name="hardware"></a>
ESP32-based LOLIN 32 MCU, MSGEQ7 Seven Band Spectrum Analyzer Breakout Board, MAX4466 Electret Microphone Amplifier, and SSD1306 OLED Display.  The schematic is presented in the diagrams/schematic.png.

## Software <a name="software"></a>
The code is separated into two sections. The models/ directory contains Python scripts to annotate and collect the data from the device (capture.py) and train the models (train.py). The device/ directory contains C++ code to generate spectrograms and to run inference. The code runs in two modes - serial I/O for acquiring training data and operational. Standard Arduino setup() and loop() methods are in the main_functions.cpp

Major Dependencies:
 
* Python 3.9
* TensorFlow 2.5.0
* PlatformIO
* TensorFlow Lite for ESP32 and PlatformIO (https://github.com/atomic14/tensorflow-lite-esp32)
* Circular Buffer Library (https://github.com/rlogiacco/CircularBuffer)
* MSGEQ7 IC Arduino Library (https://github.com/MajicDesigns/MD_MSGEQ7)
* OLED SSD1306 driver (https://github.com/ThingPulse/esp8266-oled-ssd1306)

## Roadmap <a name="roadmap"></a>

Several optimizations and enhancements may be required to make the classifier ready for testing in the field, including:
* A proximity sensor to save power (alternatively, one could wake up the classifier only when the ambient signal level is above a certain threshold).* Solar battery charger.* More sensitive directional microphone.
* Quantization for reducing memory requements.

Integration with a detering device is another, separate topic.

It will be interesting to apply the methology to other audio signals, experiment with a viraety of classification models, and to port the code to STM32   


## License <a name="license"></a>
Distributed under the MIT License. See LICENSE for more information.

## Project Status <a name="status"></a>
The work will be continued if find interest in the comunity.

## Contact <a name="contact"></a>
Alex Greysukh - agreysukh@gmail.com

Project Link: [https://github.com/good-point-lab/woodpecker-drummig-detector](https://github.com/good-point-lab/woodpecker-drummig-detector)

## Acknowledgements <a name="acknowledgements"></a>

TynyML comunity for inspiring interest in the field

Copyright (c) 2021, Alexander Greysukh where it applies