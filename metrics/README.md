
Code of two formal metrics  we introduce we introduced: Alignment and Coherence.

 The Alignment
metric assesses the semantic consistency of edits with the given prompt, ensuring
accurate modifications while preserving fidelity in the rest of the image. 

On the other hand, the Coherence metric evaluates the overall aesthetic quality of
the edited image, considering factors such as lighting and shadow consistency,
style coherence, and edge smoothnes.

Requirement:

```
pip3 install -r requirments.txt
```


Usage:

```
python eval.py --metric_type alignment/coherence --data_folder [path_to_HQ-Edit]

``` 