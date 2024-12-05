# SequenceFilter Class Documentation

A utility class that processes text chunks to filter out specific sequences of characters. This class maintains a buffer and processes incoming chunks of text to identify and remove specified sequences.

## Constructor

```dart
SequenceFilter(String sequence)
```
Creates a new filter with a specified sequence to match and remove.

## Properties

- `sequence` (`String`): The target sequence to filter out
- `buffer` (`StringBuffer`): Internal buffer for maintaining state between chunks

## Methods

### `String? processChunk(String chunk)`

Processes a chunk of text and filters out occurrences of the specified sequence.

#### Parameters
- `chunk` (`String`): The input text chunk to process

#### Returns
- `String?`: Filtered text chunk, or null if entire chunk was filtered
- Returns null if the chunk is completely consumed by the sequence matching
- Returns remaining unmatched text if a non-matching character is found

## Example Usage

```dart
// Create a filter for a specific sequence
final filter = SequenceFilter("abc");

// Process text chunks
String? result1 = filter.processChunk("ab"); // Returns null (partial match)
String? result2 = filter.processChunk("cd"); // Returns "d" (sequence broken)
String? result3 = filter.processChunk("abc"); // Returns null (complete match)
```

## Implementation Details

- Uses a rolling buffer to handle sequences split across multiple chunks
- Optimizes memory usage by only buffering potential matches
- Handles partial matches across chunk boundaries
- Clears buffer when sequence is complete or broken

## Notes

- Thread-safe for single instance use
- Maintains state between chunk processing
- Memory efficient for long sequences
- Handles both complete and partial matches

The class is particularly useful for streaming text processing where sequences may be split across multiple chunks of input text.