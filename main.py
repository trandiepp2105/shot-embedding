from shot_embedding import BatchProcessor, PipelineConfig


config = PipelineConfig(
    videos_dir="/path/to/videos",
    shots_dir="/path/to/shots",
    output_dir="/path/to/output_shot_embeddings",
    frame_step=6,
    similarity_threshold=0.88,
    pool_temperature=10.0,
    start_index=0,
    end_index=None,
    clip_model_name="ViT-H-14-quickgelu",
    clip_pretrained="dfn5b",
    batch_size=256,
    device="cuda",
    save_dtype="float16",
    overwrite=False,
)


def main():
    processor = BatchProcessor(config)
    summary = processor.run()
    print(summary)


if __name__ == "__main__":
    main()
