use rust_bert::{resources::RemoteResource, gpt_neo::{GptNeoModelResources, GptNeoConfigResources, GptNeoMergesResources}, gpt2::Gpt2VocabResources, pipelines::{text_generation::{TextGenerationConfig, TextGenerationModel}, common::ModelType}};

fn main() {
    let model_resource = Box::new(RemoteResource::from_pretrained(
        GptNeoModelResources::GPT_NEO_2_7B,
    ));

    let config_resource = Box::new(RemoteResource::from_pretrained(
        GptNeoConfigResources::GPT_NEO_2_7B,
    ));

    let vocab_resource = Box::new(RemoteResource::from_pretrained(
        Gpt2VocabResources::GPT2_MEDIUM,
    ));
    let merges_resource = Box::new(RemoteResource::from_pretrained(
        GptNeoMergesResources::GPT_NEO_2_7B,
    ));

    let generate_config = TextGenerationConfig{
        model_type: ModelType::GPTNeo,
        model_resource,
        config_resource,
        vocab_resource,
        merges_resource: Some(merges_resource),
        num_beams: 5,
        no_repeat_ngram_size: 2,
        ..Default::default()
    };

    let model = TextGenerationModel::new(generate_config).unwrap();

    loop {
        println!("Enter context and sentence");

        let mut line = String::new();
        std::io::stdin().read_line(&mut line).unwrap();

        let split = line.split('/').collect::<Vec<&str>>();
        let slc = split.as_slice();

        let output =model.generate(&slc[1..], slc[0]);

        for sentence in output{
            println!("{}", sentence);
        }

    }
}

