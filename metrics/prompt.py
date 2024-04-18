# Evaluation image changes whether consistent with the EDIT TEXT 
ALIGNMENT_PROMPT = """
From 0 to 100, how much do you rate for EDIT TEXT in terms of the correct and comprehensive description of the change from the first given image to the second given image?
Correctness refers to whether the text mentions any change that are not made between two images. 
Comprehensiveness refers to whether the text misses any change that are made between two images.
The second image should have minimum change to reflect the changes made with EDIT TEXT.
Be strict about the changes made between two images:
1. If the EDIT TEXT is about stylization or lighting change, then no content should be changed and all the details should be preserved.
2. If the EDIT TEXT is about a local change, then no irrelevant area nor image style should be changed.
3. The first image should not have the attribute described inside the EDIT TEXT, rate low, (<80) if this happens
4. Be aware to check whether the second image does maintain the important attribute in the left image that is not reflected in the EDIT TEXT. Rate low (<50) if two images are not related.
Provide a few lines for explanation and give the final response in a json format as such:
{{
    "Explanation": "",
    "Score": 30, 
}}
"""
EDIT_ITEM_EXAMPLE = """EDIT TEXT: {edit_action}"""

# Evaluation image coherence
COHERENCE_PROMPT =  """
Rate the Coherence of the provided image on a scale from 0 to 100, with 0 indicating extreme disharmony characterized by numerous conflicting or clashing elements, and 100 indicating perfect harmony with all components blending effortlessly. Your evaluation should rigorously consider the following criteria:
1. Consistency in lighting and shadows: Confirm that the light source and corresponding shadows are coherent across various elements, with no discrepancies in direction or intensity.
2. Element cohesion: Every item in the image should logically fit within the scene's context, without any appearing misplaced or extraneous.
3. Integration and edge smoothness: Objects or subjects should integrate seamlessly into their surroundings, with edges that do not appear artificially inserted or poorly blended.
4. Aesthetic uniformity and visual flow: The image should not only be aesthetically pleasing but also facilitate a natural visual journey, without abrupt interruptions caused by disharmonious elements.

Implement a stringent scoring guideline:
- Award a high score (90-100) solely if the image could pass as a flawlessly captured scene, devoid of any discernible disharmony.
- Assign a moderate to high score (70-89) if minor elements of disharmony are present but they do not significantly detract from the overall harmony.
- Give a moderate score (50-69) if noticeable disharmonious elements are evident, affecting the image's harmony to a moderate degree.
- Allocate a low score (30-49) for images where disharmonious elements are prominent, greatly disturbing the visual harmony.
- Reserve the lowest scores (0-29) for images with severe disharmony, where the elements are so discordant that it disrupts the intended aesthetic.

Your assessment must be detailed, highlighting the specific reasons for the assigned score based on the above criteria. Conclude with a response formatted in JSON as shown below:
{
    "Explanation": "<Insert detailed explanation here>",
    "Score": <Insert precise score here>
}
"""

