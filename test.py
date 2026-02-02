import torch
import os
import time
import cv2
import numpy as np
from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler
from models.T2I.pipeline import PhotoMakerStableDiffusionXLPipeline
from PIL import Image
from torchvision.transforms.functional import normalize
from models.basicsr.utils import img2tensor, tensor2img
from models.basicsr.utils.misc import gpu_is_available, get_device
from models.facelib.utils.face_restoration_helper import FaceRestoreHelper
from models.basicsr.utils.registry import ARCH_REGISTRY

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sr_device = get_device()

pretrain_model_url = {'restoration': 'checkpoint/final.pth'}


def set_realesrgan():
    from models.basicsr.archs.rrdbnet_arch import RRDBNet
    from models.basicsr.utils.realesrgan_utils import RealESRGANer
    use_half = False
    if torch.cuda.is_available():
        no_half_gpu_list = ['1650', '1660']
        if not any(gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list):
            use_half = True
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=2)
    upsampler = RealESRGANer(
        scale=2,
        model_path="D:/Desktop/TFGNet/faceSR/weights/realesrgan/RealESRGAN_x2plus.pth",
        model=model,
        tile=1600,
        tile_pad=40,
        pre_pad=0,
        half=use_half
    )
    if not gpu_is_available():
        import warnings
        warnings.warn('Runs on CPU! It will be very slow.', category=RuntimeWarning)
    return upsampler


def enhance_image(img):
    img = img.astype(np.float32)
    img = np.clip((img / 255.0) ** 0.95 * 255.0, 0, 255)
    return img.astype(np.uint8)


def resize_with_padding(img, target_size=(512, 512)):
    h, w = img.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_size = (int(w * scale), int(h * scale))
    img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
    delta_w = target_w - img_resized.shape[1]
    delta_h = target_h - img_resized.shape[0]
    top, bottom = delta_h // 2, delta_h - delta_h // 2
    left, right = delta_w // 2, delta_w - delta_w // 2
    return cv2.copyMakeBorder(
        img_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )


bg_upsampler = set_realesrgan()

sr_net = ARCH_REGISTRY.get('CodeFormer')(
    dim_embd=512,
    codebook_size=1024,
    n_head=8,
    n_layers=9,
    connect_list=['32', '64', '128', '256']
).to(sr_device)

ckpt_path = pretrain_model_url['restoration']
checkpoint = torch.load(ckpt_path, map_location=sr_device)['params_ema']
sr_net.load_state_dict(checkpoint)
sr_net.eval()

face_helper = FaceRestoreHelper(
    2,
    face_size=512,
    crop_ratio=(1, 1),
    det_model='retinaface_resnet50',
    save_ext='png',
    use_parse=True,
    device=sr_device
)


def sr_enhance_pil(pil_img):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    face_helper.clean_all()
    face_helper.read_image(img)

    num_det_faces = face_helper.get_face_landmarks_5(
        only_center_face=True,
        resize=640,
        eye_dist_threshold=5
    )

    if num_det_faces > 0:
        face_helper.align_warp_face()
        for cropped_face in face_helper.cropped_faces:
            cropped_face_t = img2tensor(
                cropped_face / 255., bgr2rgb=True, float32=True
            )
            normalize(
                cropped_face_t,
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
                inplace=True
            )
            cropped_face_t = cropped_face_t.unsqueeze(0).to(sr_device)
            try:
                with torch.no_grad():
                    output = sr_net(
                        cropped_face_t,
                        w=0.75,
                        adain=True
                    )[0]
                    restored_face = tensor2img(
                        output, rgb2bgr=True, min_max=(-1, 1)
                    )
                del output
                torch.cuda.empty_cache()
            except Exception:
                restored_face = tensor2img(
                    cropped_face_t, rgb2bgr=True, min_max=(-1, 1)
                )

            restored_face = restored_face.astype('uint8')
            restored_face = cv2.resize(
                restored_face, (512, 512),
                cv2.INTER_LINEAR
            )
            face_helper.add_restored_face(
                restored_face, cropped_face
            )

        bg_img = bg_upsampler.enhance(img, outscale=1)[0]
        face_helper.get_inverse_affine(None)
        restored_img = face_helper.paste_faces_to_input_image(
            upsample_img=bg_img,
            draw_box=False,
            face_upsampler=bg_upsampler
        )
    else:
        restored_img = img

    restored_img = resize_with_padding(restored_img, (512, 512))
    restored_img = enhance_image(restored_img)

    noise = np.random.normal(0, 1.5, restored_img.shape)
    restored_img = np.clip(
        restored_img.astype(np.float32) + noise,
        0, 255
    ).astype(np.uint8)

    return Image.fromarray(
        cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
    )


image_times = []
total_start_time = time.time()

pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
    pretrained_model_name_or_path="RealVisXL_V3.0",
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    variant="fp16"
).to(device)

pipe.tokenizer.model_max_length = 77

photomaker_path = "checkpoint/v1.bin"
pipe.load_photomaker_adapter(
    os.path.dirname(photomaker_path),
    subfolder="",
    weight_name=os.path.basename(photomaker_path),
    trigger_word="img"
)

pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config
)
pipe.fuse_lora()

input_base_folder = './test1'
output_base_folder = './out1'
info_folder = './info_en'
time_log_path = './time.txt'

os.makedirs(output_base_folder, exist_ok=True)

with open(time_log_path, 'w', encoding='utf-8') as log_file:
    log_file.write("Image number, Generation time (seconds)\n")

negative_prompt = "(low quality, illustration, 3d, painting)"
generator = torch.Generator(device=device).manual_seed(42)

for folder_name in os.listdir(input_base_folder):
    input_folder_path = os.path.join(input_base_folder, folder_name)
    if not os.path.isdir(input_folder_path):
        continue

    txt_filename = f"{folder_name[:4]}.txt"
    txt_filepath = os.path.join(info_folder, txt_filename)
    if not os.path.exists(txt_filepath):
        continue

    with open(txt_filepath, 'r', encoding='utf-8') as txt_file:
        prompt_template = txt_file.read().strip()

    image_basename_list = os.listdir(input_folder_path)
    image_path_list = sorted(
        [os.path.join(input_folder_path, b)
         for b in image_basename_list]
    )

    input_id_images = [
        sr_enhance_pil(load_image(p))
        for p in image_path_list
    ]

    start_time = time.time()

    images = pipe(
        prompt=prompt_template,
        input_id_images=input_id_images,
        negative_prompt=negative_prompt,
        num_images_per_prompt=1,
        num_inference_steps=8,
        start_merge_step=6,
        generator=generator,
    ).images[0]

    end_time = time.time()
    elapsed_time = end_time - start_time
    image_times.append(elapsed_time)

    images = images.resize((512, 512), Image.Resampling.LANCZOS)
    output_image_path = os.path.join(
        output_base_folder, f"{folder_name}.png"
    )
    images.save(output_image_path)
    print(f"Saved image for '{folder_name}' to '{output_image_path}' (Time: {elapsed_time:.2f} second)")
    with open(time_log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"{folder_name},{elapsed_time:.4f}\n")

total_time = time.time() - total_start_time
average_time = sum(image_times) / len(image_times) if image_times else 0

with open(time_log_path, 'a', encoding='utf-8') as log_file:
    log_file.write(f"\nTotal generation time (seconds): {total_time:.4f}\n")
    log_file.write(f"Total number of generated images: {len(image_times)}\n")
    log_file.write(f"Average generation time (seconds per image): {average_time:.4f}\n")

print(f"\nTime statistics have been saved to: {time_log_path}")
print(f"Total number of generated images: {len(image_times)}")
print(f"Total generation time: {total_time:.2f} second")
print(f"Average generation time (seconds per image): {average_time:.2f}")
