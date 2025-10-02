"""Standard analysis workflows.

This module defines pre-configured workflows for common analysis tasks.
These workflows can be used as templates or run directly.
"""

STANDARD_WORKFLOWS = {
    'gas_analysis': {
        'description': 'Complete gas property analysis including density, temperature, and phase diagrams',
        'plots': [
            {
                'type': 'projection',
                'field': 'Density',
                'axes': [0, 1],
                'colormap': 'algae',
                'width': 30
            },
            {
                'type': 'projection',
                'field': 'Temperature',
                'axes': [0, 1],
                'colormap': 'hot',
                'width': 30
            },
            {
                'type': 'phase',
                'x_field': 'Density',
                'y_field': 'Temperature',
                'weight_field': 'Masses',
                'limits': {
                    'x': [1e-29, 1e-21],
                    'y': [1e1, 1e7]
                }
            },
            {
                'type': 'profile',
                'x_field': 'particle_position_cylindrical_radius',
                'y_field': 'Masses',
                'bins': 50,
                'x_range': [0, 15]
            }
        ]
    },
    
    'star_formation': {
        'description': 'Star formation analysis including SFR history and KS relation',
        'plots': [
            {
                'type': 'sfr_history',
                'time_range': [0, 1000],
                'bins': 50
            },
            {
                'type': 'ks_relation',
                'radius_limit': 15
            },
            {
                'type': 'metallicity_distribution',
                'bins': 50,
                'range': [0.001, 0.1]
            }
        ]
    },
    
    'basic_comparison': {
        'description': 'Basic multi-simulation comparison plots',
        'plots': [
            {
                'type': 'projection',
                'field': 'Density',
                'axes': [0, 1],
                'colormap': 'algae',
                'width': 30
            },
            {
                'type': 'phase',
                'x_field': 'Density',
                'y_field': 'Temperature',
                'weight_field': 'Masses'
            },
            {
                'type': 'profile',
                'x_field': 'particle_position_cylindrical_radius',
                'y_field': 'particle_velocity_cylindrical_theta',
                'bins': 50
            }
        ]
    },
    
    'detailed_gas': {
        'description': 'Detailed gas properties including rotation curves and dispersion',
        'plots': [
            {
                'type': 'density_projection',
                'width': 30
            },
            {
                'type': 'temperature_projection',
                'width': 30
            },
            {
                'type': 'metallicity_map',
                'width': 30
            },
            {
                'type': 'rotation_curve',
                'max_radius': 15
            },
            {
                'type': 'velocity_dispersion',
                'max_radius': 15
            }
        ]
    },
    
    'nsff_analysis': {
        'description': 'NSFF (Near-Solar Feedback Fidelity) analysis workflow from AGORA collaboration',
        'plots': [
            {
                'type': 'surface_density_radial',
                'radius_range': [0, 15],
                'bins': 50
            },
            {
                'type': 'surface_density_vertical',
                'height_range': [0.001, 1.4],
                'bins': 10
            },
            {
                'type': 'velocity_profile',
                'radius_range': [0, 15],
                'bins': 50
            },
            {
                'type': 'velocity_dispersion',
                'radius_range': [0, 15],
                'bins': 50
            },
            {
                'type': 'density_pdf',
                'density_range': [1e-29, 1e-21],
                'bins': 50
            },
            {
                'type': 'temperature_pdf',
                'temp_range': [1e1, 1e7],
                'bins': 50
            }
        ]
    }
}
