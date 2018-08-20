# Neural Korean Person-Name Transliterator

In this project, I implemented a system to translate (transliterate) Roman alphabet person-names into Hangul person-names.
For examples,
- michael -> 마이클
- peter -> 피터  

You can check the result files in `./results` directory.  
(Roman person-name \t hangul person-name \t top-5 candidates)
## Requirements
- Python 2.7
- Tensorflow >=1.8.0
- Numpy
- tqdm
- codecs
- sys
- six
- functools
## Execution
- STEP 0, Download Roman alphabet<-> Hangul person-name dataset  
(from https://github.com/steveash/NETransliteration-COLING2018/tree/master/data)  
Data in the `./datas` directory is refined data.
- STEP 1, Adjust hyper parameters in `hyperparams.py`. Of course, you do not need to modify it!
- STEP 2, Run `python train.py` for training model. models are saved in `./logdir/` directory.
- STEP 3, Run `python inference.py` for testing model. Results are saved in the `./results/` directory.
You can also use pre-training models in `./logdir`. 
### Pre-trained models
- `./logdir/first` is pre-training model which is trained on data without refining.
- `./logdir/second` is pre-training model which is trained on data with refining. Therefore, This one may have better performance than `./logdir/first`.
## Notes
- I adapt [Tacotron](https://pdfs.semanticscholar.org/f258/f0d3260e7fbdd961993086aaafa2afc714c9.pdf) to this projects with two modicatons. First, the output
layer is modified to produce a character sequence as output, according to [neural-japanese-transliterator](https://github.com/Kyubyong/neural_japanese_transliterator). Second, CBHG module is simplified by
omitting highway network, because the length of an input sequence is much shorter than that of the original Tacotron.
- Outputs are top-5 candidates output by beam search.
- The project did not consider country adaptation. But generally, person-names have different pronunciations depending on the country. Therefore, you need to implement country adaptation for precise uses. 
- If you have a set of small country-specific datasets, you can implement country adaptation by fine-tuning the pre-training model.
(For details, see ["Country Adaptation in Neural Machine Transliteration of Person Names](https://confit.atlas.jp/guide/event-img/jsai2018/2L4-04/public/pdf?type=in))
- This project does not follow standard foreign language notation. The project follows the notation that estimates the results by neural model trained on training data.
Therefore, we have no responsibility for the uses of this project.
## References
- https://github.com/Kyubyong/neural_japanese_transliterator
- https://github.com/keithito/tacotron
- https://github.com/steveash/NETransliteration-COLING2018
