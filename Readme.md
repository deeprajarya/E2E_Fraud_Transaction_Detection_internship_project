# End to End Project Boiler Plate (Template)

## Steps for End to End Project

1.	Create a folder and open in VS code
2.	Create env and activate it
3.	Create requirements.txt
4.	Create template.py – it will create all the project structure with notebook, scr, init_setup.sh and all
5.	You can automate your env and requirements.txt file installation process
•	In init_setup.sh file , write the commands for env creation, packages installation(i.e requirements.txt)
•	Follow these commands ----
``` 
•	echo [$(date)]: "START"
•	
•	echo [$(date)]: "creating env with python 3.8 version" 
•	
•	conda create --prefix ./env python=3.8 -y
•	
•	echo [$(date)]: "activating the environment" 
•	
•	source activate ./env
•	
•	echo [$(date)]: "installing the dev requirements" 
•	
•	pip install -r requirements.txt
•	
•	echo [$(date)]: "END" 
```
•	Note – [ Some times in windows system, creating env command or activating env command does not run so try to create and activate env manually ]

6.	You have created setup.py file in above automation process. Now add command into this file.
•	"Setup.py" file is used to install local package.

•	Use following commands for installation
```
•	from setuptools import find_packages,setup
•	from typing import List
•	
•	setup(
•	    name='Fraud_TX',
•	    version='0.0.1',
•	    author='Deepraj Arya',
•	    author_email='mailforarya000@gmail.com',
•	    install_requires=["scikit-learn","pandas","numpy"],
•	    packages=find_packages()
•	)
 ```



7.	Create your own package in src folder ( like src/DiamandPricePrediction/Components/data_ingesion.py )
•	You created your local package but python will know it( i.e. you have local package )
•	For this problem python comes with this convention – create __init__.py file in every folder.
•	Python consider local package if it is with “__init__.py” file. We can say that “__init__.py” file is the way to tell python that it is my local package.

8.	Now, how to install your local package
•	Using “pip list”, You can check in terminal there will not your local package
•	To install or download global package we can use “ pip install package_name“
•	For installing your local package there are 3 ways
1.	“pip install . “   -   Here . means local package.
2.	Write “-e .” in requirements.txt
3.	Using setup.py “ python setup.py install “   -   run this file

•	You can now check using “pip list “……there must be available your local package.
•	After installing your local package successfully, you will get 3 files ---- 
1.	“package_name.egg-info” file
2.	“build”
3.	“dist”

•	If you want to publish your package, you can publish this “package_name.egg-info” file pypy repository. After publishing this package any one can use the package like we use “scikit-learn”, “pandas”, “numpy”.

9. 
