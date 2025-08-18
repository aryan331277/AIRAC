from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import uuid

with open("scraped_data_final.json", "r", encoding="utf-8") as f:
    parents = json.load(f)

for parent in parents:
    parent["parent_id"] = str(uuid.uuid4())

with open("scraped_data_final.json", "w", encoding="utf-8") as f:
    json.dump(parents, f, indent=2, ensure_ascii=False)


with open("scraped_data_final.json", "r", encoding="utf-8") as f:
    parents = list(json.load(f))

children = []
text_splitter = RecursiveCharacterTextSplitter(separators=["\n","."],chunk_size=80,chunk_overlap=20)


for parent in parents:
    title = parent["title"]
    if parent["text"] != "":
        text_chunks = text_splitter.split_text(parent["text"])
        for chunk in text_chunks:
            children.append({
                "child_id": str(uuid.uuid4()),
                "parent_id": parent["parent_id"],
                "parent_source":parent["source"],
                "parent_title":parent["title"],
                "chunk_type":"text_chunk",
                "text":f"{title} | {chunk}",
                "original_data": chunk
            })
    if parent["tables"] !=[]:
        for row in parent["tables"]:
            text = ";".join([f'"{key}"="{value}"' for key, value in row.items()])
            children.append({
                "child_id": str(uuid.uuid4()),
                "parent_id": parent["parent_id"],
                "parent_source":parent["source"],
                "parent_title":parent["title"],
                "chunk_type":"table_chunk",
                "text":f"{title} | {text}",
                "original_data": row
            })
            


with open("children_json.json", "w") as childre:
    json.dump(children, childre, indent=4)