This class acts as the primary interface for interacting with the llama.cpp library, offering a range of methods for model management, tokenization, and decoding. Here's a brief overview:

1. **Class Description**: `Llama` is designed to interface with the llama.cpp library, handling operations like loading, managing the model and context, and providing utilities for tokenization and decoding.

2. **Properties**:
   - **model**: Pointer to the Llama model.
   - **context**: Pointer to the Llama context.
   - **batch**: Configuration for the Llama model batch processing.
   - **tokensList**: List of tokens generated from the input prompt.
   - **temporaryInvalidCChars**: Storage for invalid C characters during processing.
   - **length**: Length of the output, default is -1.
   - **cursor**: Position in the token list.
   - **decode**: Counter for decoding operations.

3. **Methods**:
   - **Llama constructor**: Loads the model and context based on provided parameters.
   - **dispose()**: Releases all allocated resources.
   - **modelInfo()**: Returns information about the model.
   - **setPrompt(String prompt)**: Sets the prompt for the model and prepares it for processing.
   - **getNext()**: Generates and returns the next token in the sequence.
   - **prompt(String prompt)**: Provides a stream of generated text based on the given prompt.
   - **clear()**: Clears the current token list and temporary data.
   - **batchAdd()**: Utility method for adding tokens to the batch.
   - **tokenize()**: Tokenizes a given text string.
   - **tokenToPiece()**: Converts a token ID to its corresponding piece (text representation).