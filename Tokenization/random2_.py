import regex as re
import tiktoken

# gpt2pat = re.compile(
#     r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# )

# # print(re.findall(gpt2pat, "Hello've world123 how are you"))



# # Fixed the variable name from 'nc' to 'enc'
# enc = tiktoken.get_encoding("gpt2") 
# print(enc.encode("Hello World ")) 

# # This block works correctly
# enc2 = tiktoken.get_encoding("cl100k_base") 
# print(enc2.encode("Hello World "))

import regex as re
import unicodedata
# s = "Konnichiwa (こんにちは)"
# # print([ord(x) for x in s])
# print(list("Konnichiwa (こんにちは)".encode("utf-8")))

text = "The unexceptionable synchronization of their ultra-modernized cryptocurrency algorithms unexpectedly failed during the multi-dimensional simulation, causing 10,450 automated supercomputers to repeatedly recalculate the unquantifiable data—unbelievable, right? 🤔 Failure-driven experimentation ultimately builds unstoppable, unshakeable stability."

tokens = text.encode("utf-8")
print(list(map(int,tokens)))