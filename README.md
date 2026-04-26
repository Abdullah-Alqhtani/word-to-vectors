# word-to-vectors!
the result of the code


RUNNING
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Repo card metadata block was not found. Setting CardData to empty.
Dataset({
    features: ['tweet', 'label'],
    num_rows: 47000
})
Dataset({
    features: ['tweet', 'label'],
    num_rows: 11751
})
{'tweet': 'اعترف ان بتس كانو شوي شوي يجيبو راسي لكن اليوم بالزايد 😭', 'label': 0}
Example:
[['اعترف', 'ان', 'بتس', 'كانو', 'شوي', 'شوي', 'يجيبو', 'راسي', 'لكن', 'اليوم', 'بالزايد'], ['توقعت', 'اذا', 'جات', 'داريا', 'بشوفهم', 'كاملين', 'بس', 'لي', 'للحين', 'احس', 'فيه', 'احد', 'ناقصهم'], ['الاهلي', 'الهلال', 'اكتب', 'توقعك', 'لنتيجة', 'لقاء', 'الهلال', 'والاهلي', 'تحت', 'التاق', 'تحدي', 'اسرع', 'روقان', 'وادخل', 'في', 'سحب', 'قيمة', 'ايفون', 'على']]
Training Word2Vec...
Word2Vec training finished
Vocabulary size: 32894
Test word: اعترف
Vector shape: (100,)
Most similar words:
[('وحدة', 0.9773716926574707), ('تغير', 0.9757311940193176), ('الأخ', 0.9744874238967896), ('حلم', 0.9743896722793579), ('بقوته', 0.9740321636199951)]
Model saved successfully
Creating sentence vectors...
Train vectors shape: (47000, 100)
Train labels shape: (47000,)
Test vectors shape: (11751, 100)
Test labels shape: (11751,)
Epoch 1, Loss: 0.6897664666175842
Epoch 2, Loss: 0.6868299245834351
Epoch 3, Loss: 0.684091329574585
Epoch 4, Loss: 0.6815108060836792
Epoch 5, Loss: 0.6790398359298706
Accuracy: 0.5775678665645477
