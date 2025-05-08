import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class SMRM(torch.nn.Module):
    def __init__(self, mask_decoder, adaptive_prompt_generator):
        super(SegmentationMaskRefinementModule, self).__init__()
        self.mask_decoder = mask_decoder
        self.adaptive_prompt_generator = adaptive_prompt_generator

    def compute_cosine_similarity(self, pixel_features, prompt_features):
        """
        Compute the cosine similarity between each pixel feature and each prompt feature.
        Args:
            pixel_features: Tensor of shape (M, D) where M is the number of pixels and D is the feature dimension.
            prompt_features: Tensor of shape (N, D) where N is the number of prompts and D is the feature dimension.
        Returns:
            Cosine similarity matrix of shape (M, N).
        """
        pixel_norm = F.normalize(pixel_features, p=2, dim=-1)  # Normalize pixel features
        prompt_norm = F.normalize(prompt_features, p=2, dim=-1)  # Normalize prompt features
        
        # Compute cosine similarity
        similarity_matrix = torch.mm(pixel_norm, prompt_norm.t())
        return similarity_matrix

    def compute_adaptive_threshold(self, cost_matrix):
        """
        Compute the adaptive threshold based on the average of the cost matrix.
        Args:
            cost_matrix: Tensor of shape (M, N) containing the cost matrix.
        Returns:
            Adaptive threshold value.
        """
        threshold = cost_matrix.mean()
        return threshold

    def forward(self, pixel_features, prompt_features):
        """
        Forward pass that computes the refined segmentation mask and bounding boxes.
        Args:
            pixel_features: Tensor of shape (M, D), where M is the number of pixels and D is the feature dimension.
            prompt_features: Tensor of shape (N, D), where N is the number of prompts and D is the feature dimension.
        Returns:
            refined_mask: The refined segmentation mask of shape (M,).
            refined_bboxes: Corresponding bounding boxes after refinement.
        """
        # Step 1: Compute cosine similarity (cost matrix)
        cost_matrix = self.compute_cosine_similarity(pixel_features, prompt_features)
        
        # Step 2: Compute the adaptive threshold
        tau = self.compute_adaptive_threshold(cost_matrix)
        
        # Step 3: Apply Hungarian algorithm to find the minimum cost matching
        cost_matrix_np = cost_matrix.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(-cost_matrix_np)  # Maximizing the similarity
        
        # Step 4: Assign pixel labels based on the Hungarian matching
        refined_mask = torch.full((pixel_features.size(0),), -1, dtype=torch.long)  # Initialize with background (-1)
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] >= tau:
                refined_mask[i] = j  # Assign pixel to prompt j
            else:
                refined_mask[i] = -1  # Background if below threshold
        
        # Step 5: Generate refined bounding boxes (optional, depends on specific use case)
        refined_bboxes = self.generate_refined_bboxes(refined_mask)
        
        return refined_mask, refined_bboxes
    
    @torch.no_grad()
    def generate_refined_bboxes(self, refined_mask):
        """
        Generate refined bounding boxes from the segmentation mask.
        Args:
            refined_mask: The refined segmentation mask of shape (M,).
        Returns:
            refined_bboxes: List of bounding boxes for each cluster of pixels (prompt).
        """
        # Example placeholder for bounding box generation logic.
        # In a real implementation, this would find the bounding box for each unique prompt label.
        refined_bboxes = []
        unique_labels = refined_mask.unique()
        for label in unique_labels:
            if label != -1:  # Skip background label
                mask = refined_mask == label
                # Get bounding box coordinates (xmin, ymin, xmax, ymax)
                xmin, ymin = mask.nonzero().min(dim=0)[0]
                xmax, ymax = mask.nonzero().max(dim=0)[0]
                refined_bboxes.append((xmin.item(), ymin.item(), xmax.item(), ymax.item()))
        return refined_bboxes
