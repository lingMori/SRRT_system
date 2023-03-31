import os

work_dir = os.path.dirname(os.path.abspath(__file__))
print(work_dir)

os.chdir("streamlit_fix")

os.system('streamlit run main.py')
