Here's a concise overview:

1. **Enumeration `LlamaSplitMode`**:
   - **Purpose**: This enum defines the modes for splitting a Llama model across multiple GPUs, dictating how the model's computational load is distributed.
   - **Values**:
     - **none**: Indicates usage of a single GPU for the model.
     - **layer**: Specifies that layers and Key-Value (KV) pairs are split across multiple GPUs.
     - **row**: Designates that rows of the model are distributed across different GPUs.