# FloatingFarmYaw

## Brief Summary
This repository contains the code for the manuscript "Floating Offshore Wind Farm Yaw Control via Model-based Deep Reinforcement Learning," accepted by the IEEE Power & Energy Society General Meeting 2025. The authors are Mingyang Mei, Peng Kou, Yilin Xu, Zhihao Zhang, Runze Tian, and Deliang Liang, with Peng Kou as the corresponding author.

An extended version of this work, titled "[Improving Floating Offshore Wind Farm Flow Control with Scalable Model-based Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/11062486)" has been accepted for publication in _IEEE Transactions on Automation Science and Engineering_.

For more information about the repository authors:

Mingyang Mei: [Google Scholar](https://scholar.google.com/citations?user=jpXmO2UAAAAJ&hl=zh-CN)
Peng Kou: [Xian Jiaotong University Homepage](https://gr.xjtu.edu.cn/en/web/koupeng)

## Illustrations and visualizations:
<div align=center>
     <img src="Results/illustration.png" height="175"/> 
</div>
<div align=center>
      Fig.1 The illustration of floating offshore wind turbine repositioning (a) A two-turbine array configuration (b) Top view of the turbine repositioning with yaw control
</div>

<div align=center>
     <img src="Results/Wind Farm.png" height="480"/> 
</div>
<div align=center>
      Fig.2 Simulation results (a) Time-averaged model with yaw control (b) Fast.Farm model with yaw control (c) Time-averaged model without yaw control (d) Fast.Farm model without yaw control.
</div>


## Requirements
This repository is dependent on [Floris v4](https://github.com/NREL/floris), [MoorPy v1.0](https://github.com/NREL/MoorPy), [Pytorch v2.4](https://pytorch.org/) and field measurements from [Pywake](https://github.com/DTUWindEnergy/PyWake). If someone wants to deploy the trained agent, [onnx](https://onnx.ai/) will also be required for converting the deep neural networks.


## Quick Use

1. clone the repository
```pycon
git clone https://github.com/yangmingmei/FloatingFarmYaw.git
```
2. Install the dependency using pip
```pycon
cd FloatingFarmYaw
pip install . -e
```
3. Install PyTorch (you must choose the right CUDA version):
See: https://pytorch.org/get-started/locally/. This step is optional because Pytorch is only necessary for DRL training.

4. run [mooring_matrix.py](mooring_matrix.py) to see the mooring configurations.
   
5. run [floris_environment.py](floris_environment.py) to see the iteration process of the model.

6. run [main.py](main.py) to train a DRL agent for floating wind farm yaw control (it only take 3 hours when using massively parallel simulations). 

## Acknowledgement
This work was supported by the National Natural Science Foundation of China under Grant 52077165. (Principle investigator: [Peng Kou](https://gr.xjtu.edu.cn/en/web/koupeng))

## Future development
This code repository is currently not a final release and under development. Documentations, as well as the codes for validation on FAST.Farm, will be released. Moreover, the code is fully compatible with the anual energy production and layout optimization method used in FLORIS. This allows the users to further explore its potantial.


https://github.com/user-attachments/assets/e78fd109-8298-4f4e-9a46-8ca7bcad6fa8

<div align=center>
    Validation on a utility-scale floating wind farm with water depth of 400~450 m 
</div>

## License
This project is licensed under the terms of the [Apache License Version 2.0](LICENSE)
