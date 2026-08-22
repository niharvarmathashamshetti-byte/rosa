import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from skimage import measure
import math

"""
Volume Visualizer Module
------------------------
This module provides visualization functions for the ROSA Knee AI project. 
It supports both CT intensity volumes and discrete segmentation label maps.

Medical Imaging Context:
- Axis 0 (Z-axis) typically corresponds to the Axial plane (top-to-bottom).
- Axis 1 (Y-axis) typically corresponds to the Coronal plane (front-to-back).
- Axis 2 (X-axis) typically corresponds to the Sagittal plane (left-to-right).
"""

def _handle_save_and_show(save_path: str | None = None, show: bool = True, close_fig: bool = False) -> None:
    """Helper function to manage saving, showing, and closing matplotlib figures."""
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    if close_fig:
        plt.close()

def _get_middle_index(volume: np.ndarray, axis: int) -> int:
    """Helper to safely get the middle index of a volume along a specific axis."""
    if volume.size == 0:
        return 0
    return volume.shape[axis] // 2

def _get_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    """Helper to extract a 2D slice from a 3D volume given an axis and index."""
    if axis == 0:
        return volume[index, :, :]
    elif axis == 1:
        return volume[:, index, :]
    elif axis == 2:
        return volume[:, :, index]
    else:
        raise ValueError("Axis must be 0, 1, or 2.")

def show_slice(
    volume: np.ndarray, 
    axis: int = 0, 
    index: int | None = None, 
    title: str = '', 
    cmap: str = 'gray', 
    vmin: float | None = None, 
    vmax: float | None = None, 
    figsize: tuple[int, int] = (8, 8),
    save_path: str | None = None,
    show: bool = True,
    close_fig: bool = False
) -> None:
    """
    Display a single 2D slice from a 3D volume.
    
    Why we need it:
    Provides a quick look at a single cross-section of the knee, which is the 
    foundation of reviewing CT scans and masks.
    
    Args:
        volume (np.ndarray): 3D volume data (CT intensities or labels).
        axis (int): 0=axial, 1=coronal, 2=sagittal.
        index (int | None): Slice index to display. If None, uses the middle slice.
        title (str): Title for the plot.
        cmap (str): Colormap to use (default 'gray' for CT).
        vmin, vmax: Min/Max values for colormap scaling.
        figsize: Figure dimension.
        save_path (str | None): If provided, saves the plot to this path.
        show (bool): If True, calls plt.show().
        close_fig (bool): If True, closes the figure to free memory.
    """
    if volume.size == 0:
        print("Warning: Empty volume provided to show_slice.")
        return

    if index is None:
        index = _get_middle_index(volume, axis)
        
    index = max(0, min(index, volume.shape[axis] - 1))  # Clamp index

    slice_2d = _get_slice(volume, axis, index)
    
    # Map axis integers to anatomical plane names
    plane_names = {0: "Axial (Z)", 1: "Coronal (Y)", 2: "Sagittal (X)"}
    plane = plane_names.get(axis, "UNKNOWN — NEEDS VERIFICATION")

    plt.figure(figsize=figsize)
    plt.imshow(slice_2d, cmap=cmap, vmin=vmin, vmax=vmax)
    
    full_title = f"{title} | {plane} Plane | Slice: {index}/{volume.shape[axis]-1}"
    plt.title(full_title)
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.colorbar(label='Intensity / Label Value')
    plt.tight_layout()
    
    _handle_save_and_show(save_path, show, close_fig)

def show_three_views(
    volume: np.ndarray, 
    indices: tuple[int, int, int] | None = None, 
    title: str = '', 
    cmap: str = 'gray', 
    vmin: float | None = None, 
    vmax: float | None = None, 
    figsize: tuple[int, int] = (18, 6),
    save_path: str | None = None,
    show: bool = True,
    close_fig: bool = False
) -> None:
    """
    Display axial, coronal, and sagittal views side by side.
    
    Why we need it:
    Radiologists and engineers need to see all three anatomical planes simultaneously 
    to understand 3D structures and verify alignment.
    
    Args:
        volume (np.ndarray): 3D array of the scan/mask.
        indices: Tuple of (axial_idx, coronal_idx, sagittal_idx). Uses middle if None.
        title, cmap, vmin, vmax, figsize, save_path, show, close_fig: Standard plotting kwargs.
    """
    if volume.ndim != 3:
        raise ValueError("Volume must be a 3D numpy array.")
        
    if indices is None:
        indices = (
            _get_middle_index(volume, 0),
            _get_middle_index(volume, 1),
            _get_middle_index(volume, 2)
        )
        
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title if title else "Three-Plane View", fontsize=16)
    
    planes = [
        (0, "Axial (Z)", indices[0]),
        (1, "Coronal (Y)", indices[1]),
        (2, "Sagittal (X)", indices[2])
    ]
    
    for ax, (axis, plane_name, idx) in zip(axes, planes):
        idx = max(0, min(idx, volume.shape[axis] - 1))
        slice_2d = _get_slice(volume, axis, idx)
        
        im = ax.imshow(slice_2d, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{plane_name} | Slice: {idx}/{volume.shape[axis]-1}")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    _handle_save_and_show(save_path, show, close_fig)

def show_slice_overlay(
    image: np.ndarray, 
    mask: np.ndarray, 
    axis: int = 0, 
    index: int | None = None, 
    alpha: float = 0.4, 
    num_labels: int | None = None, 
    figsize: tuple[int, int] = (18, 6),
    save_path: str | None = None,
    show: bool = True,
    close_fig: bool = False
) -> None:
    """
    Display 3 panels: image alone, mask alone (colored), and overlay.
    
    Why we need it:
    Crucial for qualitative evaluation of segmentation models. Allows us to see 
    exactly where the predicted labels land on the patient's anatomy.
    
    Args:
        image (np.ndarray): 3D volume (e.g., CT scan).
        mask (np.ndarray): 3D segmentation mask (categorical).
        axis (int): Viewing axis (0=Z, 1=Y, 2=X).
        index (int | None): Slice index to show.
        alpha (float): Transparency for the overlay.
        num_labels (int | None): If None, inferred from mask max value.
        figsize, save_path, show, close_fig: Standard plotting arguments.
    """
    if image.shape != mask.shape:
        raise ValueError(f"Image shape {image.shape} and mask shape {mask.shape} must match.")
        
    if index is None:
        index = _get_middle_index(image, axis)
    
    index = max(0, min(index, image.shape[axis] - 1))
    
    img_slice = _get_slice(image, axis, index)
    mask_slice = _get_slice(mask, axis, index)
    
    # Infer max labels if not provided
    if num_labels is None:
        num_labels = int(np.max(mask)) + 1
        
    # Use tab20 for distinct colors, suitable for multi-class segmentation
    cmap_mask = plt.get_cmap('tab20', num_labels)
    # Mask out the background (label 0) for the overlay so it's transparent
    masked_mask_slice = np.ma.masked_where(mask_slice == 0, mask_slice)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f"Overlay View | Axis: {axis} | Slice: {index}", fontsize=16)
    
    # Panel 1: Image only
    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_title("CT Image")
    axes[0].axis('off')
    
    # Panel 2: Mask only
    im_mask = axes[1].imshow(mask_slice, cmap=cmap_mask, vmin=-0.5, vmax=num_labels-0.5, interpolation='nearest')
    axes[1].set_title("Segmentation Mask")
    axes[1].axis('off')
    
    # Panel 3: Overlay
    axes[2].imshow(img_slice, cmap='gray')
    axes[2].imshow(masked_mask_slice, cmap=cmap_mask, alpha=alpha, vmin=-0.5, vmax=num_labels-0.5, interpolation='nearest')
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    
    # Add a legend for the labels. Skipping 0 (background).
    # NOTE: Exact anatomical meanings are UNKNOWN — NEEDS VERIFICATION. 
    legend_patches = [
        mpatches.Patch(color=cmap_mask(i), label=f'Label {i}') 
        for i in range(1, num_labels)
    ]
    if legend_patches:
        # Place legend outside the overlay plot
        axes[2].legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    _handle_save_and_show(save_path, show, close_fig)

def show_multi_slice(
    volume: np.ndarray, 
    axis: int = 0, 
    num_slices: int = 9, 
    cmap: str = 'gray', 
    figsize: tuple[int, int] = (16, 16),
    save_path: str | None = None,
    show: bool = True,
    close_fig: bool = False
) -> None:
    """
    Display a grid of slices evenly spaced through the volume along the given axis.
    
    Why we need it:
    Scanning through a single view doesn't give the whole picture. A grid allows 
    us to quickly inspect the entire 3D volume at a glance to spot anomalies or artifacts.
    
    Args:
        volume (np.ndarray): 3D volume.
        axis (int): Axis to slice along (0, 1, or 2).
        num_slices (int): Number of slices to display (best if a perfect square, e.g., 9, 16).
        cmap, figsize, save_path, show, close_fig: Standard plotting args.
    """
    max_idx = volume.shape[axis] - 1
    # Generate evenly spaced indices
    indices = np.linspace(0, max_idx, num_slices, dtype=int)
    
    grid_size = math.ceil(math.sqrt(num_slices))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < num_slices:
            idx = indices[i]
            slice_2d = _get_slice(volume, axis, idx)
            ax.imshow(slice_2d, cmap=cmap)
            ax.set_title(f"Slice: {idx}")
            ax.axis('off')
        else:
            ax.axis('off')  # Hide unused subplots
            
    plt.tight_layout()
    _handle_save_and_show(save_path, show, close_fig)

def show_label_distribution(
    volume: np.ndarray, 
    title: str = 'Label Distribution',
    save_path: str | None = None,
    show: bool = True,
    close_fig: bool = False
) -> None:
    """
    Show bar chart and table of voxel counts/percentages per label for a segmentation mask.
    
    Why we need it:
    Essential for understanding class imbalance in segmentation datasets. Also serves as 
    a diagnostic tool to ensure expected anatomy is actually present in the mask.
    
    Args:
        volume (np.ndarray): 3D segmentation mask.
        title (str): Title for the plot.
        save_path, show, close_fig: Standard plotting args.
    """
    unique_labels, counts = np.unique(volume, return_counts=True)
    total_voxels = volume.size
    
    print("-" * 40)
    print(f"{'Label':<10} | {'Voxel Count':<15} | {'Percentage':<10}")
    print("-" * 40)
    
    percentages = []
    for label, count in zip(unique_labels, counts):
        pct = (count / total_voxels) * 100
        percentages.append(pct)
        print(f"{label:<10} | {count:<15} | {pct:.4f}%")
    print("-" * 40)
    print("NOTE: Label meanings are UNKNOWN — NEEDS VERIFICATION.")

    plt.figure(figsize=(10, 5))
    bars = plt.bar(unique_labels, counts, color='skyblue')
    plt.xlabel('Label Value')
    plt.ylabel('Voxel Count')
    plt.title(title)
    plt.xticks(unique_labels)
    plt.yscale('log')  # Log scale often needed because background (0) dominates
    
    # Add percentage text on top of bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{pct:.1f}%', ha='center', va='bottom', rotation=45)

    plt.tight_layout()
    _handle_save_and_show(save_path, show, close_fig)

def show_intensity_histogram(
    volume: np.ndarray, 
    bins: int = 256, 
    title: str = 'Intensity Histogram', 
    figsize: tuple[int, int] = (10, 4),
    save_path: str | None = None,
    show: bool = True,
    close_fig: bool = False
) -> None:
    """
    Plot histogram of all voxel values to distinguish CT data from segmentation labels.
    
    Why we need it:
    CT data has a wide, continuous range of values (Hounsfield Units), while masks 
    have a few discrete values (e.g., 0, 1, 2). This helps automatically verify the data type.
    
    Args:
        volume (np.ndarray): 3D volume.
        bins (int): Number of histogram bins.
        title (str): Title of the plot.
        figsize, save_path, show, close_fig: Standard plotting args.
    """
    # Flatten the array to 1D for histogram
    flat_data = volume.ravel()
    
    mean_val = np.mean(flat_data)
    median_val = np.median(flat_data)
    min_val = np.min(flat_data)
    max_val = np.max(flat_data)
    
    plt.figure(figsize=figsize)
    plt.hist(flat_data, bins=bins, color='gray', alpha=0.7)
    
    # Mark statistics
    plt.axvline(mean_val, color='r', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='b', linestyle='dashed', linewidth=1.5, label=f'Median: {median_val:.2f}')
    
    plt.title(f"{title}\nMin: {min_val:.2f} | Max: {max_val:.2f}")
    plt.xlabel("Voxel Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.yscale('log') # Log scale because air/background usually heavily outweighs bone/tissue
    
    plt.tight_layout()
    _handle_save_and_show(save_path, show, close_fig)

def show_3d_preview(
    volume: np.ndarray, 
    labels_to_show: list[int] | None = None, 
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0), 
    figsize: tuple[int, int] = (10, 10),
    max_size: int = 128,
    save_path: str | None = None,
    show: bool = True,
    close_fig: bool = False
) -> None:
    """
    Basic 3D surface rendering of a segmentation mask. NOT for production quality.
    
    Why we need it:
    Sanity check to ensure the segmentation masks form coherent 3D anatomical structures 
    (like a femur or tibia) and aren't just scattered noise.
    
    Args:
        volume (np.ndarray): 3D segmentation mask.
        labels_to_show (list[int]): Specific labels to render. If None, renders all > 0.
        spacing (tuple): Voxel spacing (Z, Y, X) for physical scaling.
        figsize (tuple): Figure dimension.
        max_size (int): Max dimension size. Volumes larger than this will be downsampled to save RAM.
        save_path, show, close_fig: Standard plotting args.
    """
    # Create copy or downsample to avoid out-of-memory errors on large volumes
    vol_shape = volume.shape
    step = max(1, max(vol_shape) // max_size)
    
    if step > 1:
        print(f"Downsampling volume by factor {step} for 3D preview to fit max_size {max_size}.")
        vol_render = volume[::step, ::step, ::step]
        # Adjust spacing based on downsample factor
        render_spacing = (spacing[0]*step, spacing[1]*step, spacing[2]*step)
    else:
        vol_render = volume
        render_spacing = spacing

    if labels_to_show is None:
        labels_to_show = [lbl for lbl in np.unique(vol_render) if lbl > 0]
        
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.get_cmap('tab10', len(labels_to_show))
    
    rendered_any = False
    
    for idx, label in enumerate(labels_to_show):
        binary_mask = (vol_render == label)
        if not np.any(binary_mask):
            print(f"Warning: Label {label} has no voxels in the (downsampled) volume.")
            continue
            
        try:
            # Marching cubes extracts a 2D surface mesh from a 3D volume
            verts, faces, normals, values = measure.marching_cubes(binary_mask, level=0.5, spacing=render_spacing)
            
            # plot_trisurf draws the extracted mesh
            ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], 
                            color=colors(idx), alpha=0.5, edgecolor='none', label=f'Label {label}')
            rendered_any = True
        except Exception as e:
            print(f"Failed to render label {label}. Error: {e}")
            
    if rendered_any:
        ax.set_title("3D Segmentation Preview")
        ax.set_xlabel(f"Z-axis (spacing={render_spacing[0]})")
        ax.set_ylabel(f"Y-axis (spacing={render_spacing[1]})")
        ax.set_zlabel(f"X-axis (spacing={render_spacing[2]})")
        
        # A simple hack to support legends in mplot3d
        legend_elements = [mpatches.Patch(color=colors(i), label=f'Label {lbl}') 
                           for i, lbl in enumerate(labels_to_show)]
        ax.legend(handles=legend_elements)
    else:
        ax.text(0.5, 0.5, 0.5, "No labels to render", ha='center')

    plt.tight_layout()
    _handle_save_and_show(save_path, show, close_fig)


def show_each_label_slice(
    volume: np.ndarray, 
    axis: int = 0, 
    index: int | None = None, 
    figsize: tuple[int, int] = (20, 4),
    save_path: str | None = None,
    show: bool = True,
    close_fig: bool = False
) -> None:
    """
    Show a separate subplot for each unique label's binary mask at a given slice.
    
    Why we need it:
    When a single slice has multiple overlapping or closely adjacent labels (e.g. femur, 
    tibia, patella), viewing them in isolation helps confirm their boundaries without visual clutter.
    
    Args:
        volume (np.ndarray): 3D segmentation mask.
        axis (int): Axis to slice.
        index (int | None): Slice index.
        figsize, save_path, show, close_fig: Standard plotting args.
    """
    if index is None:
        index = _get_middle_index(volume, axis)
        
    index = max(0, min(index, volume.shape[axis] - 1))
    slice_2d = _get_slice(volume, axis, index)
    
    unique_labels = [lbl for lbl in np.unique(slice_2d) if lbl > 0]
    
    if not unique_labels:
        print(f"No positive labels found in slice {index} along axis {axis}.")
        return
        
    num_labels = len(unique_labels)
    fig, axes = plt.subplots(1, num_labels, figsize=(max(figsize[0], 4*num_labels), figsize[1]))
    
    # Handle single subplot edge case gracefully
    if num_labels == 1:
        axes = [axes]
        
    fig.suptitle(f"Isolated Labels | Axis: {axis} | Slice: {index}", fontsize=14)
    
    for ax, label in zip(axes, unique_labels):
        binary_mask = (slice_2d == label)
        ax.imshow(binary_mask, cmap='gray')
        ax.set_title(f"Label {label}")
        ax.axis('off')
        
    plt.tight_layout()
    _handle_save_and_show(save_path, show, close_fig)
