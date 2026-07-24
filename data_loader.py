from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser


def load_data(path):
    loader = PyPDFLoader(path, mode="page")
    data = loader.load()
    return data


if __name__ == "__main__":
    data = load_data("The_Stranger_Albert_Camus-removed-removed.pdf")
    print(data[0].page_content)

