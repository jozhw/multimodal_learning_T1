import os
import openslide
import pandas as pd

slides_dir = '/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/preprocessing/TCGA_WSI/LUAD_all/svs_files/FFPE_single_sample_per_patient/'
output_dir = '/lus/eagle/clone/g2/projects/GeomicVar/tarak/multimodal_learning_T1/preprocessing/png_files/'

def save_slide_as_png(slide, slide_path):
    thumbnail = slide.get_thumbnail((1024, 1024))  # Create a thumbnail
    png_filename = os.path.join(output_dir, os.path.basename(slide_path).replace('.svs', '.png'))
    thumbnail.save(png_filename)
    return png_filename


def get_slide_stats(slide_path):
    try:
        slide = openslide.OpenSlide(slide_path)
        mpp_x = slide.properties.get('openslide.mpp-x')
        mpp_y = slide.properties.get('openslide.mpp-y')
        base_magnification = slide.properties.get('openslide.objective-power')

        if base_magnification is not None:
            base_magnification = float(base_magnification)

        downsample_factors = slide.level_downsamples
        magnifications = [base_magnification / ds if base_magnification else None for ds in downsample_factors]

        png_path = save_slide_as_png(slide, slide_path)

        slide_stats = {
            'File': os.path.basename(slide_path),
            'MPP X': mpp_x,
            'MPP Y': mpp_y,
            'Base Magnification': base_magnification,
            'Width': slide.dimensions[0],
            'Height': slide.dimensions[1],
            'Levels': slide.level_count,
            'Downsamples': downsample_factors,
            'Tile Width': slide.level_dimensions[0][0],
            'Tile Height': slide.level_dimensions[0][1],
            'Magnifications': magnifications,
            'PNG Path': png_path
        }
        slide.close()
        print(slide_stats)
        return slide_stats
    except openslide.OpenSlideError as e:
        return {'File': os.path.basename(slide_path), 'Error': str(e)}
    except ValueError as e:
        return {'File': os.path.basename(slide_path), 'Error': 'MPP metadata is missing'}


def get_all_slides_stats(slides_dir):
    stats = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for root, _, files in os.walk(slides_dir):
        for file in files:
            if file.endswith('.svs'):
                slide_path = os.path.join(root, file)
                stats.append(get_slide_stats(slide_path))
    return stats


def save_stats_to_csv(slide_stats, filename='slide_statistics.csv'):
    df = pd.DataFrame(slide_stats)
    df.to_csv(filename, index=False)
    print(f"Slide statistics saved to {filename}")


# get stats for all slides and save them to a CSV file
slide_stats = get_all_slides_stats(slides_dir)
save_stats_to_csv(slide_stats)