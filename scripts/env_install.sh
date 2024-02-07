conda create -n minigpt5 -y python=3.9.18
source activate minigpt5
pip install -r requirements.txt
pip install filelock==3.9.0
pip install salesforce-lavis==1.0.2
pip install git+https://github.com/facebookresearch/detectron2.git@8c4a333ceb8df05348759443d0206302485890e0
cd src/Ego4d && pip install .