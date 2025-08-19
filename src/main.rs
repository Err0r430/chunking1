use async_openai::types::{CreateChatCompletionRequest, ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage, ChatCompletionRequestSystemMessageContent, ChatCompletionRequestUserMessage};
use async_openai::{Client, config::OpenAIConfig};
use async_openai::error::OpenAIError;
use tiktoken_rs::o200k_base;
use std::fs::File;
use std::io::{Read, Write};
use dotenv;
use std::env;
use futures::future::join_all;

pub fn get_env(key: &str) -> String {
    env::var(key).unwrap_or_else(|_| panic!("Environment variable {} not set", key))
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() {
    dotenv::dotenv().ok();
    // I'm doing this system because this is a learning project. Nothing production grade and I'm aware of that.
    let key = get_env("OPENAI_KEY");

    let config = OpenAIConfig::new().with_api_key(key);

    let client = Client::with_config(config);

    // Generic function from another project that I merged into what I neded. Named wrong and probably can be combined with to_shakespeare...
    // However, this allows for some flexibility 🤷‍♂️ nitpicky.

    // Scaff func for getting ai responses.
    pub async fn get_completion_response(client: Client<OpenAIConfig>, sys: &str, prompt: (&str, &str, &str)) -> Result<String, OpenAIError> {
        let chat = client.chat();
        let response = chat.create(CreateChatCompletionRequest {
            // Smarter models probably work better with this, originally tried 4o-mini and it was faster but stupid.
            model: "gpt-4o".to_owned(),
            messages: vec![
                ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                    content: ChatCompletionRequestSystemMessageContent::Text(sys.to_string()),
                    name: None,
                }),
                ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage { // Idea for this was to get across to the model what the start and end of each block was. I do have a concern for injection. Not quite sure the best way to do that.
                    content: async_openai::types::ChatCompletionRequestUserMessageContent::Text(format!("START CONTEXT CONTENT {} END CONTEXT CONTENT", prompt.0.to_string())),
                    name: None,
                }),
                ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage { // Same as above, but for the content.
                    content: async_openai::types::ChatCompletionRequestUserMessageContent::Text(format!("START CONTENT {} END CONTENT", prompt.1.to_string())),
                    name: None,
                }),
                ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage { // Same as above, but for the suffix.
                    content: async_openai::types::ChatCompletionRequestUserMessageContent::Text(format!("START SUFFIX CONTENT {} END SUFFIX CONTENT", prompt.2.to_string())),
                    name: None,
                }),
            ],
            // I'm not sure at all how much effect this has. I read that lowering it to 0 can increase the chance of getting a good response. Might be wrong and worth investigating.
            temperature: Some(0.0),
            ..Default::default()
        }).await?;

        Ok(response.choices[0].message.content.clone().unwrap_or_else(|| "LLM failed to respond.".to_string()))
    }

    // Wrapper around the completion function. Just to make it easier. Ended up swapping from shakespeare to british english. Was funnier.
    pub async fn to_shakespeare(client: Client<OpenAIConfig>, context: &str, content: &str, suffix: &str) -> Option<String> {
        get_completion_response(client,
            r#"You will be provided 3 user messages.
            The first message is prepending context, something to be aware of whats before.
            The middle is actual content.
            The final is more context post content.
            Turn the middle, which is the content, into sophisticated english speak.
            Proper words, best vocab, the air of british gentlemen.
            Do not rewrite the context/suffix they are only for your knwoledge."#, (context, content, suffix)).await.ok()
        // I am unsure if this is the ideal prompting method!
    }

    // Now this takes a bit of thought so bear with me here.
    /*

    So heres what im thinking. Before this, I had no real clue what a sliding window was but i realize now this is that after some googling.
    No "im so smart and invented something" for me :(

    On that heres my gameplan.
    Our first step is we need to know the model's context window. We arent making something model agnostic. (its possible with this though)
    Now we know the context window we have to figure out how much of that window our text takes. Thankfully tiktoken has rs bindings so it was a breeze.
    now we have 2 constant bits of info. How much the model can handle, and how much our content is.
    With that we come to our first task, splitting it up.
    My initial idea was to grab a bunch of different slices one after the other and just send them to get processed at the same time.
    I never ended up doing that and went with one after the other. Wait that original might still work what.

    EDIT: Alright gents its been 5 mins and I refactored my code to spawn them to run concurrently. Its now around 200ms faster and thats well worth it.

    On that side note, the slicing.
    We have all the essential parts of the equation with us. Context limit, content size.

    What I did next was just (content size / lmit) and that was number of chunks.
    I then took each of those chunks and processed them with ai.
    It worked! But it had a major flaw. If it split mid sentence, it just repaired it. Even when instructed not to its flow was off.

    So then I read more into sliding window and found about the overlap idea.
    My equation was then shifted to our total chunks being:
    content size / limit - overlap amount.ceil()
    (dummy nums) ( 10_000 / (500 - 50) ).ceil(). or 23

    This seemed to work much better and is what I ended up doing.

    So yeah it looks like I have little to 0 overlap on my content. I'm happy with this!
    I would love some feedback though please let me know.

     */
    const CONTEXT_LIMIT: usize = 500;

    const OVERLAP_MARGIN: usize = 50; // 200 token overlap was too much originally.

    const CHUNK_SIZE: usize = CONTEXT_LIMIT - OVERLAP_MARGIN;

    let mut file = File::open("bee.txt").unwrap(); // Ideally in prod we have a failsafe
    let mut contents = String::new(); // Establish contents
    let _ = file.read_to_string(&mut contents); // Mutate file to be its contents now.

    let tk = o200k_base().unwrap(); // Init tokenizer

    // Turn it into tokens
    let tokens = tk.encode_ordinary(&contents);

    // Get how many tokens there are
    let token_length = tokens.len();
    println!("There are {} tokens", token_length);

    // Get how many chunks there will be. Tokens / chunks.
    let total_chunks = (token_length.clone() + CHUNK_SIZE - 1) / CHUNK_SIZE;

    println!("total of {} chunks to go through", total_chunks);


    println!("Starting to process..");

    // Spawn tasks
    let mut tasks: Vec<tokio::task::JoinHandle<Option<String>>> = Vec::new();
    for i in 0..total_chunks {
        // Clone the client so we can use it in the task.
        let client = client.clone();
        println!("processing chunk {}", i);

        // Content start and end.
        let start = i * CHUNK_SIZE;
        let end = (start + CHUNK_SIZE).min(token_length.clone());

        // Context start (plain start is the end)
        let context_start = start.saturating_sub(OVERLAP_MARGIN);

        // Suffex end (plain end is the start)
        let suffix_end = end.saturating_add(OVERLAP_MARGIN).clamp(0, token_length);

        // actual content chunk
        let content_chunk = &tokens[start..end]; // This is our current chunk slice, a list of the tokens.

        // Context chunk
        let context_chunk = &tokens[context_start..start];

        // Suffix chunk
        let suffix_chunk = &tokens[end..suffix_end];


        // Context text.
        let context_text = tk.decode(context_chunk.to_vec()).unwrap();

        // Content text
        let content_text = tk.decode(content_chunk.to_vec()).unwrap();

        // Suffix text
        let suffix_text = tk.decode(suffix_chunk.to_vec()).unwrap();


        println!("shakespearing...");
        tasks.push(tokio::spawn(async move {
            to_shakespeare(client, &context_text, &content_text, &suffix_text).await
        }));
    }

    let results = join_all(tasks).await;

    // collect into final script
    let mut tweaked_script = String::new();
    for r in results {
        if let Ok(Some(txt)) = r {
            tweaked_script.push_str(&txt);
        }
    }
    println!("Finished: {}", tweaked_script);

    let mut file = File::create("shakesbee.txt").unwrap();
    file.write_all(tweaked_script.as_bytes());
}
