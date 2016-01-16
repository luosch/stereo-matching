##Language
c++11
##Dependencies
```
cmake
opencv
```
##Generate the results
```
cmake .
make
./runner
```
##Directory structure
```
.
├── README.md
├── main.cpp (program entry)
├── StereoMatching.hpp (implement SSD, NCC and ASW)
├── ALL-2views (source image)
└── result (output image)
```
##original views
![](https://raw.githubusercontent.com/luosch/stereo-matching/master/ALL-2views/Aloe/view1.png)
![](https://raw.githubusercontent.com/luosch/stereo-matching/master/ALL-2views/Aloe/view5.png)
##SSD
![](https://raw.githubusercontent.com/luosch/stereo-matching/master/result/Aloe_disp1_SSD.png)
![](https://raw.githubusercontent.com/luosch/stereo-matching/master/result/Aloe_disp1_SSD.png)
##NCC
![](https://raw.githubusercontent.com/luosch/stereo-matching/master/result/Aloe_disp1_NCC.png)
![](https://raw.githubusercontent.com/luosch/stereo-matching/master/result/Aloe_disp1_NCC.png)
##ASW
![](https://raw.githubusercontent.com/luosch/stereo-matching/master/result/Aloe_disp1_ASW.png)
![](https://raw.githubusercontent.com/luosch/stereo-matching/master/result/Aloe_disp1_ASW.png)
## License
This code is distributed under the terms and conditions of the MIT license.