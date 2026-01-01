It's a successful demo for learning text2video big model.<br>
Base on model:DFloat11/Wan2.2-T2V-A14B-2-DF11<br>
https://huggingface.co/DFloat11/Wan2.2-T2V-A14B-2-DF11<br>
modelscope download Wan-AI/Wan2.2-T2V-A14B-Diffusers --local_dir ./Wan-AI/Wan2.2-T2V-A14B-Diffusers <br>
modelscope download DFloat11/Wan2.2-T2V-A14B-2-DF11 --local_dir ./DFloat11/Wan2.2-T2V-A14B-2-DF11<br>
modelscope download DFloat11/Wan2.2-T2V-A14B-DF11 --local_dir ./DFloat11/Wan2.2-T2V-A14B-DF11<br>
uv pip install dfloat11[cuda12] imageio imageio-ffmpeg<br>

Need minimum VRAM 22GB,128GB RAM

