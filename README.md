# TXM_Sandbox
This is a GUI program based on Jupyter that integrates tomography reconstruction and 2D/3D XANES data analysis. It provides a mechanism that allows users to customize the input data file formats.


Installation
  via conda-forge

	conda create -n pytxm python=3.10 txm_sandbox -c conda-forge
	conda activate pytxm


Usage
  You also need to install Fiji/ImageJ in your computer. After installing Fiji, you will need to update ImageJ then Fiji from its 'Help' menu. Restart Fiji after the updates.

  You will need to copy TXM_GUI2.ipynb into a working directory. Change to that working directory, then run

	jupyter lab
  
  This will launch Jupyter in your preferred web browser. The first cell in the opened Jupyter notebook is for launching TXM_Sandbox GUI. The second line

	gui = txmg.txm_gui(fiji_path='/home/xiao/software/Fiji.app', form_sz=[750, 1000])

includes the Fiji path as its argument. Change the path to your Fiji installation path. Click on the first cell and do shift+enter will run this cell. This will bring up TXM_Sandbox GUI. 


References:
  Please cite these two papers if you use TXM_Sandbox in your works.

1. Xiao, X., Xu, Z., Lin, F. & Lee, W.-K., "TXM-Sandbox: an open-source software for transmission X-ray microscopy data analysis" (2022). J. Synchrotron Rad. 29, 266-275. https://doi.org/10.1107/S1600577521011978

2. Xiao, X., Xu, Z., Hou, D., Yang, Z. & Lin, F., "Rigid registration algorithm based on the minimization of the total variation of the difference map" (2022). J. Synchrotron Rad. 29, 1085-1094. https://doi.org/10.1107/S1600577522005598
 
