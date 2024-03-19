import os 
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp

####################################################

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
n_batch = 16  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Template definitions
template = """Question: {question}

Answer: Let's get straight to the point."""

prompt = PromptTemplate.from_template(template)

template2 = """Question: {question}

Answer: Let's work this out in a step-by-step way to be sure we have the right answer."""

prompt2 = PromptTemplate.from_template(template2)

conclusion1 = """Question: {question}

Answer: Let's summarize this text."""

fin1 = PromptTemplate.from_template(conclusion1)

conclusion2 = """Question: {question}

Answer: Pick the most popular opinion from the given list."""

fin2 = PromptTemplate.from_template(conclusion2)

# Model paths - Make sure the model path is correct for your system!
LLM = LlamaCpp(
    model_path="./llama-2-7b-chat.Q4_0.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    max_tokens=15,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

LLM2 = LlamaCpp(
    model_path="./llama-2-7b-chat.Q4_0.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    max_tokens=1000,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

##############################################################

num_of_bots = int(input("\nEnter the number of bots: "))
user_prompt = input("\nEnter the prompt:\n")
output_list = []

# Counters for different answers
yes_count = 0
no_count = 0
it_depends_count = 0

# Create a text prompt
prompt_text = user_prompt + " Give a short definitive answer to this question based on your knowledge."

llm_chain = LLMChain(prompt=prompt, llm=LLM)

# Generate responses
for i in range(num_of_bots):
    output = llm_chain.run(prompt_text)
    print(output)
    output_list.append(output)
    output_list.append("\n")

print("\n\n///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////\n")
result = ", ".join(output_list)
print(result)

prompt2_text = result + " Find the most popular statement in this text."
llm_chain = LLMChain(prompt=fin1, llm=LLM2)

print("\n\n///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////\n")
output_final = llm_chain.run(prompt2_text)
print("\n")
