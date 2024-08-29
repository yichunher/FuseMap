import os, warnings

warnings.filterwarnings("ignore")

# from ..config import ModelType
from typing import Tuple, Union
from langchain_community.llms import VLLM
from langchain_community.llms import VLLMOpenAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI, AzureOpenAIEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

#
from langchain_community.chat_models.azureml_endpoint import AzureMLChatOnlineEndpoint, AzureMLEndpointApiType, LlamaChatContentFormatter

import argparse
from enum import Enum
from typing import Dict, Any

def get_key(model):
    AZURE_OPENAI_API_VERSION = "2023-12-01-preview"


    # API_KEY_SC="fea17040f840483eb0c577e7db513122"
    # API_KEY_ES2="98261b5b86414be989ecb4aac4e1b429"
    # API_KEY_GENIE = "fakeKey"

    # if model in [
    #     "gpt-35-turbo-16k",
    #     "dall-e-3",
    #     "gpt-4",
    #     "gpt-4-32k",
    #     "gpt-4-vision-preview",
    #     "text-embedding-ada-002",
    # ]:
    #     openai_api_key = API_KEY_SC
    #     url = f"https://gred-project-g-sweden-central-test.openai.azure.com/"  # openai/deployments/{model}/chat/completions?api-version=2023-12-01-preview"

    # elif model in [
    #     "gpt-4o",
    #     "gpt-35-turbo",
    #     "text-embedding-3-large",
    #     "text-embedding-3-small",
    # ]:
    #     url = f"https://rnd-chat-eastus2-test.openai.azure.com/"  # openai/deployments/{model}/chat/completions?api-version=2023-12-01-preview"
    #     openai_api_key = API_KEY_ES2

    # elif model == "genie":
    #     url = "http://localhost:8000/v1"
    #     openai_api_key = "stagent-rocks"

    # else:
    #     raise NotImplementedError

    AZURE_OPENAI_API_KEY_EUS2="da5f7e6c846340ebb094bd9ecbd542e8"
    AZURE_OPENAI_API_KEY_SC="3bc8e8e74d1841dd98785b2d8b09ae9e"

    if model in [
        "gpt-35-turbo",
        "gpt-4",
        "gpt-4o",
        "gpt-4-vision-preview",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    ]:
        openai_api_key = AZURE_OPENAI_API_KEY_SC
        url = f"https://regevlab-swedencentral-test.openai.azure.com/"  # openai/deployments/{model}/chat/completions?api-version=2023-12-01-preview"

    elif model in [
        "text-embedding-3-small",
    ]:
        url = f"https://regevlab-eastus2-test.openai.azure.com/"  # openai/deployments/{model}/chat/completions?api-version=2023-12-01-preview"
        openai_api_key = AZURE_OPENAI_API_KEY_EUS2
    else:
        raise EOFError

    return openai_api_key, url, AZURE_OPENAI_API_VERSION


def make_llm(model, temp, streaming: bool = False):
    api_key, url, AZURE_OPENAI_API_VERSION = get_key(model)

    if model == "genie":
        # llm = GenieLLM(api_key=api_key, base_url=url, cookies=GenieCookies)
        llm = VLLMOpenAI(
            openai_api_key="stagent-rocks",
            openai_api_base="http://localhost:8000/v1",
            model_name="models/geniemodels.dev.gcs.gene.com/models_hf/Meta-Llama-3-8B-Instruct",
        )
        
        # llm = VLLMOpenAI(
        #     openai_api_key="stagent-rocks",
        #     openai_api_base="http://localhost:7890/v1",
        #     model_name="models/mosaicml/mpt-7b",
        # )
    else:
        llm = AzureChatOpenAI(
            openai_api_key=api_key,
            azure_endpoint=url,
            azure_deployment=model,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            temperature=temp,
            max_tokens=4000,
            # request_timeout=1000,
            streaming=streaming,
            callbacks=[StreamingStdOutCallbackHandler()],
        )
    return llm


def make_llm_emb(emb_model):
    api_key, url, AZURE_OPENAI_API_VERSION = get_key(emb_model)
    os.environ["AZURE_OPENAI_API_KEY"] = api_key

    if emb_model == "genie":
        raise NotImplementedError
        # cookies = GenieCookies
        # embeddings = GenieEmbeddings(api_key=api_key, base_url=url, emb_model=emb_model, cookies=cookies)
    else:
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=emb_model,
            azure_endpoint=url,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            # chunk_size=1
        )

    return embeddings
