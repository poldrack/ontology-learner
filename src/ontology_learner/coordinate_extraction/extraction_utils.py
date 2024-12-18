def get_coord_prompt(table_data):
    return f"""
The following text contains the data from a table in a published research paper that uses 
functional MRI brain imaging.  The table contains coordinates of brain activation, 
listed as X/Y/Z locations, as well as labels for the anatomical location of those regions.  
Please extract the locations (which should be numbers in the range of -150 to 150) and labels.  If possible, 
please also extract the type of statistic being reported (e.g. t, z), the value of the statistic 
(which should be a number), the cluster size, and the type of coordinate (e.g., MNI, Talairach).  

There may be multiple tables that contain information about different comparisons between
conditions, known as contrasts (e.g. "Incongruent > Congruent"). 

## TABLE DATA ##
{table_data}

## RESPONSE ##

You should return a JSON list, with no additional text. If there are
no coordinates present in the table, return an empty list.

The output should include list of dictionaries for each coordinate in the table, each containing the following keys:
    - contrast: what concept or comparison between conditions is being reported (e.g. "Incongruent > Congruent", "Correlation with age")
    - x: the x coordinate (a number between -150 and 150)
    - y: the y coordinate (a number between -150 and 150)
    - z: the z coordinate (a number between -150 and 150)
    - cluster_size: the size of the cluster (a number)
    - label: the label for the anatomical location (e.g. "Left hemisphere")
    - statistic_type: the type of statistic being reported (e.g. t, Z, r)
    - statistic_value: the value of the statistic (a number)
    - coordinate_type: the stereotaxic coordinate system used (usually MNI or Talairach)
    
"""

def get_sorted_coords(a_coord, b_coord):
    a_coord = a_coord.sort_values(by=['x','y','z'])
    b_coord = b_coord.sort_values(by=['x','y','z'])
    return(a_coord, b_coord)