from zf_pf_diffeo.pipeline import do_referenceGeometries,do_temporalreferenceGeometries


import logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # Define folder paths
    proj_dir = "/media/max_kotz/sahra_shivani_data/sorted_data/morphoMaps/projected_surfaces"
    maps_dir = "/media/max_kotz/sahra_shivani_data/sorted_data/morphoMaps/Maps"
    temp_maps_dir = "/media/max_kotz/sahra_shivani_data/sorted_data/morphoMaps/Maps_temp"
    # Run processing
    do_referenceGeometries(proj_dir,["genotype","condition","time in hpf"],maps_dir,"projected_data","Projected Surface file name")
    do_temporalreferenceGeometries(maps_dir, "time in hpf", ["genotype","condition"], temp_maps_dir)
