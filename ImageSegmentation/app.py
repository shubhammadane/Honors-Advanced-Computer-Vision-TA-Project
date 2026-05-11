import gradio as gr

from src.inference import BackgroundRemover


def build_ui() -> gr.Interface:
    remover = BackgroundRemover()

    def remove_bg(image, output_mode):
        if image is None:
            return None
        return remover.remove(image, output_mode=output_mode)

    return gr.Interface(
        fn=remove_bg,
        inputs=[
            gr.Image(type="pil", label="Input Image"),
            gr.Radio(
                ["transparent", "white"],
                value="transparent",
                label="Output Background",
            ),
        ],
        outputs=gr.Image(type="pil", label="Cutout"),
        title="Background Removal",
        description="Upload an image and remove the background using a pre-trained segmentation model.",
    )


def main() -> None:
    ui = build_ui()
    ui.launch()


if __name__ == "__main__":
    main()