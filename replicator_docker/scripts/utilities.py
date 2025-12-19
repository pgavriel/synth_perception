import os 
import json
import math
from typing import Optional
from pxr import Sdf, Usd


def export_prim_to_layer(prim: Usd.Prim, flatten=True,
                         include_session_layer=True) -> Optional[Sdf.Layer]:
    """
    Creates a temporary layer with only the given prim in it

    Parameters
    ----------
    prim: Usd.Prim
        The usd object we wish to export by itself into its own layer
    flatten: bool
        If True, then the returned layer will have all composition arcs
        flattened. If False, then the returned layer will contain a reference
        to the original prim.
    include_session_layer: bool
        If True, then include changes from the session layer. If this is
        False, and the prim is ONLY defined in the session layer, then None
        will be returned.

    Note that no parents are included, and xforms are not "flattened" (even if
    `flatten` is True - `flatten` refers to USD composition arcs, not parent
    hierarchy or xforms).
    """
    orig_stage = prim.GetStage()
    orig_root_layer = orig_stage.GetRootLayer()
    orig_session_layer = orig_stage.GetSessionLayer()
    if orig_session_layer and not include_session_layer:
        # Make sure that the prim still exists if we exclude the session layer
        stage_no_session = Usd.Stage.Open(orig_root_layer, sessionLayer=None)
        if not stage_no_session.GetPrimAtPath(prim.GetPrimPath()):
            return None

        orig_session_layer = None

    # If there is a session layer, to get an EXACT copy including possible
    # modifications by the session layer, we need to create a new "copy" layer,
    # with a layer stack composed of the orig_stage's session layer
    # and root layer
    if not orig_session_layer:
        copy_layer = orig_root_layer
    else:
        copy_layer = Sdf.Layer.CreateAnonymous()
        copy_layer.subLayerPaths.append(orig_session_layer.identifier)
        copy_layer.subLayerPaths.append(orig_root_layer.identifier)

    # Now create a "solo" stage, with only our prim (from the copy layer)
    # referenced in
    solo_stage = Usd.Stage.CreateInMemory()
    solo_prim_path = Sdf.Path(f"/{prim.GetName()}")
    solo_prim = solo_stage.DefinePrim(solo_prim_path)
    solo_prim.GetReferences().AddReference(copy_layer.identifier,
                                           prim.GetPrimPath())
    solo_stage.SetDefaultPrim(solo_prim)

    if flatten:
        return solo_stage.Flatten()
    else:
        return solo_stage.GetRootLayer()


def export_prim_to_string(prim: Usd.Prim, flatten=True,
                          include_session_layer=True) -> Optional[str]:
    """
    Return a string representation of the given prim

    Parameters
    ----------
    prim: Usd.Prim
        The usd object we wish to convert to a string
    flatten: bool
        If True, then return a representation with all USD composition arcs
        flattened. If False, then return the prim definition from the
        strongest composition arc that contributes opinions to this prim.
    include_session_layer: bool
        If True, then include changes from the session layer. If this is
        False, and the prim is ONLY defined in the session layer, then None
        will be returned.

    Note that no parents are included, and xforms are not "flattened" (even if
    `flatten` is True - `flatten` refers to USD composition arcs, not parent
    hierarchy or xforms).
    """
    if not flatten:
        prim_stack = prim.GetPrimStack()
        if not include_session_layer:
            non_session_layers = \
                prim.GetStage().GetLayerStack(includeSessionLayers=False)
            prim_stack = \
                [x for x in prim_stack if x.layer in non_session_layers]
            if not prim_stack:
                return None
            return prim.GetPrimStack()[0].GetAsText()

    solo_layer = export_prim_to_layer(
        prim, flatten=True, include_session_layer=include_session_layer)
    if solo_layer is None:
        # If include_session_layer was False, it's possible that the prim
        # doesn't exist any more...
        return None
    solo_primspec = solo_layer.GetPrimAtPath(solo_layer.defaultPrim)
    return solo_primspec.GetAsText()


def print_prim(prim: Usd.Prim, flatten=True, include_session_layer=True):
    """
    Print a string representation of the given prim.

    Parameters
    ----------
    prim: Usd.Prim
        The usd object we wish to convert to a string
    flatten: bool
        If True, then print a representation with all USD composition arcs
        flattened. If False, then print the prim definition from the
        strongest composition arc that contributes opinions to this prim.
    include_session_layer: bool
        If True, then include changes from the session layer. If this is
        False, and the prim is ONLY defined in the session layer, then 'None'
        will be printed.

    Note that no parents are included, and xforms are not "flattened" (even if
    `flatten` is True - `flatten` refers to USD composition arcs, not parent
    hierarchy or xforms).
    """
    print(export_prim_to_string(prim, flatten=flatten,
                                include_session_layer=include_session_layer))
    

import inspect
def inspect_object(obj):
    # Use inspect.getmembers() to see all attributes and their values
    print(f"Inspecting Python object: {type(obj)}")
    for name, value in inspect.getmembers(obj):
        # Optional: filter out private/protected attributes and methods
        if not name.startswith('_') and not inspect.ismethod(value):
            print(f"  {name}: {value}")

# Assuming 'stage' is your Usd.Stage object
def get_all_prim_paths(stage: Usd.Stage, verbose=True) -> list[str]:
    if verbose: print("Finding all prim paths...")
    prim_paths = []
    for prim in stage.Traverse():
        prim_paths.append(str(prim.GetPath()))
    if verbose: 
        for p in prim_paths: print(f" > {p}")
    return prim_paths

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def create_incremental_dir(root, prefix="test", digits=3):
    os.makedirs(root, exist_ok=True)  # Ensure root exists
    i = 1
    while True:
        new_dir = os.path.join(root, f"{prefix}_{i:0{digits}}")
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            return new_dir
        i += 1

def load_json_config(file, verbose=False):
    with open(file, "r") as f:
        config = json.load(f)

    if verbose:
        print("\nLoaded config:")
        print(json.dumps(config,sort_keys=False,indent=2))
        print("\n")
    
    return config

def save_json(out_file, json_dict):
    with open(out_file, "w") as f:
        json.dump(json_dict, f, indent=4)  # 'indent' makes it nicely formatted

def compute_plane_size(fov_deg, aspect_ratio, depth):
    fov_rad = math.radians(fov_deg)
    height = 2 * depth * math.tan(fov_rad / 2)
    width = height * aspect_ratio
    return width, height

def print_object_info(object,object_name=None):
    print(f"Object Info: {object_name}")
    print(f"Type: {type(object)}")
    print(f"Dir: {dir(object)}")