# Facial-Recognition-C-

Developed on May 22, 2023 By Lance Gabriel Trias

Overview: This is a finals project for my freshman year. Built using the C++ programming language, OpenCV library, and Visual Studio 2022. The project was made in a span of 2 weeks, with over a month of researching and reading similar projects across the web. While the project works overall, the only issue was a failure to implement proper trainer and facial recognition capabilities. There were more documentation and projects of OpenCV from python so it was difficult to replicate the same projects into C++.

Program: At startup, a simple GUI using forms pops up, this GUI is for the user to select either facial recognition or to take live samples for trainer. Selecting either one will open 2 kinds of your live camera frame, one is for grayscale and the other is default.
For training:
The window frame will begin a facial detection and continuously checks for 2 eyes and a face. Once the critera are met, the program will take samples of the live feed into a pre-defined folder. If the folder does not exist, it will create one.

For Recognition:
The window frame will begin a facial detection and then facial recognition using the saved samples and its corresponding owner/label. Although the program was not able to implement these features, it somewhat reached this level but was bugged and couldn't be fixed at the time.
