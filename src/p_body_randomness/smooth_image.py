'''
Smoothens the images which contain the protein signal.
'''

import cv2

def smooth_protein_image(protein_signal, pbody_mask, cell_mask):
    '''
    Function: Substracts the p-body signals from the image and smoothens it out.
    Input:    Protein signal (numpy array), p-body segmentation (numpy array), cell mask (numpy array)
    Output:   The smoothened image.
    '''
    # We first substract some signal in areas where p-bodies are located.
    no_p_bodies = 1 * protein_signal
    no_p_bodies[pbody_mask != 0] = no_p_bodies[pbody_mask != 0] * 0.75
    
    # Now we go through five rounds of smoothing the resulting image.
    smooth_array = 1 * no_p_bodies
    for i in range(5):
        smooth_array = cv2.GaussianBlur(smooth_array, (9, 9), 0)
    
    # Finally, restricting signal to area within the cell mask.
    final_image = 1 * smooth_array
    final_image[cell_mask == 0] = 0
    
    return final_image
