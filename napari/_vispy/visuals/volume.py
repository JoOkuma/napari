from __future__ import annotations

import numpy as np
from vispy.scene.visuals import Volume as BaseVolume

from napari._vispy.visuals.util import TextureMixin

FUNCTION_DEFINITIONS = """
// the tolerance for testing equality of floats with floatEqual and floatNotEqual
const float equality_tolerance = 1e-8;

bool floatNotEqual(float val1, float val2)
{
    // check if val1 and val2 are not equal
    bool not_equal = abs(val1 - val2) > equality_tolerance;

    return not_equal;
}

bool floatEqual(float val1, float val2)
{
    // check if val1 and val2 are equal
    bool equal = abs(val1 - val2) < equality_tolerance;

    return equal;
}


// the background value for the iso_categorical shader
const float categorical_bg_value = 0;

int detectAdjacentBackground(float val_neg, float val_pos)
{
    // determine if the adjacent voxels along an axis are both background
    int adjacent_bg = int( floatEqual(val_neg, categorical_bg_value) );
    adjacent_bg = adjacent_bg * int( floatEqual(val_pos, categorical_bg_value) );
    return adjacent_bg;
}

vec3 computeNormal(vec3 loc, vec3 step)
{
    float radius = 2.5f;
    float radius_step = 1.0f;
    float val0 = colorToVal($get_data(loc + step));
    vec3 N = vec3(0.0f);

    for (float z = -radius; z <= radius; z += radius_step) {
        for (float y = -radius; y <= radius; y += radius_step) {
            for (float x = -radius; x <= radius; x += radius_step)
            {
                vec3 dif = step * vec3(z, y, x);
                float val1 = colorToVal($get_data(loc + dif));
                float dif_length = length(dif);  // Could be optimized
                N = N + dif * (val0 - val1) / dif_length;
            }
        }
    }
    return normalize(N);
}

vec4 calculateShadedCategoricalColor(vec4 betterColor, vec3 loc, vec3 step)
{
    // View direction
    vec3 V = normalize(view_ray);
    vec3 N = computeNormal(loc, step);

    // Init colors
    vec4 ambient_color = vec4(0.0);
    vec4 diffuse_color = vec4(0.0);
    vec4 specular_color = vec4(0.0);

    // FIXME: testing lighting
    vec4 m_ambient = vec4(1.0, 0.0, 0.0, 1.0);
    vec4 m_diffuse = vec4(0.0, 1.0, 0.0, 1.0);
    vec4 m_specular = vec4(1.0, 1.0, 1.0, 1.0);
    vec4 final_color;

    float ka = 0.8f;
    float kd = 0.4f;
    float ks = 0.2f;
    float ns = 5.0f;

    // todo: allow multiple light, define lights on viewvox or subscene
    int nlights = 1;
    for (int i=0; i<nlights; i++)
    {
        // Get light direction (make sure to prevent zero division)
        vec3 L = normalize(view_ray);  //lightDirs[i];
        float lightEnabled = float( length(L) > 0.0 );
        L = normalize(L+(1.0-lightEnabled));

        // Calculate lighting properties
        float lambertTerm = clamp(dot(N,L), 0.0, 1.0 );

        // cos(2x) = 2 cos^2(x) - 1
        // https://en.wikipedia.org/wiki/List_of_trigonometric_identities
        float cos20 = 2 * lambertTerm * lambertTerm - 1;
        float specularTerm = pow(cos20, ns);

        // Calculate mask
        float mask1 = lightEnabled;

        // Calculate colors
        ambient_color +=  mask1 * ka * m_ambient;  // * gl_LightSource[i].ambient;
        diffuse_color +=  mask1 * kd * lambertTerm * m_diffuse;
        specular_color += mask1 * ks * specularTerm, 5.0 * m_specular;  // * gl_LightSource[i].specular;
    }

    // Calculate final color by componing different components
    final_color = betterColor * ( ambient_color + diffuse_color + specular_color);
    final_color.a = betterColor.a;
    final_color = clamp(final_color, 0.0, 1.0);

    // Done
    return final_color;
}
"""

ISO_CATEGORICAL_SNIPPETS = {
    'before_loop': """
        vec4 color3 = vec4(0.0);  // final color
        vec3 dstep = 1.5 / u_shape;  // step to sample derivative, set to match iso shader
        gl_FragColor = vec4(0.0);
        bool discard_fragment = true;
        vec4 label_id = vec4(0.0);
        float threshold = 0.01f;
        float small_step_threshold = 0.005f;
        """,
    'in_loop': """
        // check if value is different from the background value
        // almost the same as 1.0 (internal boundary) but not exactly
        if ( abs(val - 1.0f) < threshold ) {
            // Take the last interval in smaller steps
            vec3 iloc = loc - step;
            for (int i=0; i<10; i++) {
                label_id = $get_data(iloc);
                color = sample_label_color(label_id.r);
                float ival = colorToVal(label_id);
                // if (floatNotEqual(color.a, 0) ) {
                if (floatNotEqual(color.a, 0) && abs(ival - 1.0) < small_step_threshold) {
                    // fully transparent color is considered as background, see napari/napari#5227
                    // when the value mapped to non-transparent color is reached
                    // calculate the shaded color (apply lighting effects)
                    color = calculateShadedCategoricalColor(color, iloc, dstep);
                    gl_FragColor = color;

                    // set the variables for the depth buffer
                    frag_depth_point = iloc * u_shape;
                    discard_fragment = false;

                    iter = nsteps;
                    break;
                }
                iloc += step * 0.1;
            }
        }
        """,
    'after_loop': """
        if (discard_fragment)
            discard;
        """,
}

TRANSLUCENT_CATEGORICAL_SNIPPETS = {
    'before_loop': """
        vec4 color3 = vec4(0.0);  // final color
        gl_FragColor = vec4(0.0);
        bool discard_fragment = true;
        vec4 label_id = vec4(0.0);
        """,
    'in_loop': """
        // check if value is different from the background value
        if ( floatNotEqual(val, categorical_bg_value) ) {
            // Take the last interval in smaller steps
            vec3 iloc = loc - step;
            for (int i=0; i<10; i++) {
                label_id = $get_data(iloc);
                color = sample_label_color(label_id.r);
                if (floatNotEqual(color.a, 0) ) {
                    // fully transparent color is considered as background, see napari/napari#5227
                    // when the value mapped to non-transparent color is reached
                    // calculate the color (apply lighting effects)
                    gl_FragColor = color;

                    // set the variables for the depth buffer
                    frag_depth_point = iloc * u_shape;
                    discard_fragment = false;

                    iter = nsteps;
                    break;
                }
                iloc += step * 0.1;
            }
        }
        """,
    'after_loop': """
        if (discard_fragment)
            discard;
        """,
}

shaders = BaseVolume._shaders.copy()
before, after = shaders['fragment'].split('void main()')
shaders['fragment'] = before + FUNCTION_DEFINITIONS + 'void main()' + after

rendering_methods = BaseVolume._rendering_methods.copy()
rendering_methods['iso_categorical'] = ISO_CATEGORICAL_SNIPPETS
rendering_methods['translucent_categorical'] = TRANSLUCENT_CATEGORICAL_SNIPPETS


class Volume(TextureMixin, BaseVolume):
    # add the new rendering method to the snippets dict
    _shaders = shaders
    _rendering_methods = rendering_methods


class SDFVolume(Volume):
    def set_data(
        self,
        vol: np.ndarray,
        clim: tuple | None = None,
        copy: bool = True,
    ) -> None:
        try:
            from edt import sdf
        except ImportError as e:
            raise ImportError(
                'The "edt" package is required to use SDFVolume'
            ) from e
        vol_sdf = sdf(vol).astype(vol.dtype)  # FIXME: should be float
        self.interpolation = 'linear'
        super().set_data(vol_sdf, clim, copy)
