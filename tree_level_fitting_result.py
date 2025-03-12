import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib.colors as mcolors
import function_kaon as tb
List = [
    ["B451", 500, 0.136981, 0.136409, 64,  [10, 12, 14,],    0.075],
    ["B452", 400, 0.137045, 0.136378, 64,  [10, 12, 14, 16], 0.075],
    ["N450", 280, 0.137099, 0.136353, 128, [10, 12, 14, 16], 0.075],
    ["N304", 420, 0.137079, 0.136665, 128, [15, 18, 21, 24], 0.049],
    ["N305", 250, 0.137025, 0.136676, 128, [15, 18, 21, 24], 0.049],
]
conf_names = ["B451", "B452", "N450", "N304", "N305"]
kappa_hs = ["0.104000","0.115000","0.124500"]
dir = "kaon_result/"
fminv_to_GEV = 1/5.068
# Load the fit results of kaon mass
kaon_mass = {}
for i_conf, conf_name in enumerate(conf_names):
    kaon_mass[conf_name] = {}
    file_name = dir + conf_name + "GEVP_OP_fitting_result.pickle"
    with open(file_name, 'rb') as f:
        fitting_result = pickle.load(f)
    OPfitting_exponential = fitting_result["exponential"]
    fitting = OPfitting_exponential[0].best_fit
    kaon_mass[conf_name]["mass"] = fitting.res[0]*fminv_to_GEV/(List[i_conf][6])
    kaon_mass[conf_name]["mass_err"] = tb.Bootstrap_erro(np.array(fitting.boots_res)[:,0])*fminv_to_GEV/(List[i_conf][6])
    kaon_mass[conf_name]["a2"] = List[i_conf][6]**2

plt.figure(figsize=(8, 5))

for conf_name in conf_names:
    mass = kaon_mass[conf_name]["mass"]
    mass_err = kaon_mass[conf_name]["mass_err"]
    a2 = kaon_mass[conf_name]["a2"]
    
    plt.errorbar(a2, mass, yerr=mass_err, fmt="o", label=conf_name, capsize=5)

# Labels and formatting
plt.xlabel("$a^2$ (${fm}^2$)")
plt.ylabel("Kaon Mass (GeV)")
plt.title("Kaon Mass vs. $a^2$ for Different Ensemble")
plt.legend()
plt.grid()
plt.xlim((0,0.008))
# Show plot
plt.show()

kaon_mass = {}
for i_conf, conf_name in enumerate(conf_names):
    kaon_mass[conf_name] = {}
    file_name = dir + conf_name + "GEVP_OP_fitting_result.pickle"
    with open(file_name, 'rb') as f:
        fitting_result = pickle.load(f)
    OPfitting_exponential = fitting_result["exponential"]
    fitting = OPfitting_exponential[1].best_fit
    kaon_mass[conf_name]["mass"] = fitting.res[0]*fminv_to_GEV/(List[i_conf][6])
    kaon_mass[conf_name]["mass_err"] = tb.Bootstrap_erro(np.array(fitting.boots_res)[:,0])*fminv_to_GEV/(List[i_conf][6])
    kaon_mass[conf_name]["a2"] = List[i_conf][6]**2

plt.figure(figsize=(8, 5))

for conf_name in conf_names:
    mass = kaon_mass[conf_name]["mass"]
    mass_err = kaon_mass[conf_name]["mass_err"]
    a2 = kaon_mass[conf_name]["a2"]
    
    plt.errorbar(a2, mass, yerr=mass_err, fmt="o", label=conf_name, capsize=5)

# Labels and formatting
plt.xlabel("$a^2$ (${fm}^2$)")
plt.ylabel("Kaon Mass (p = 1) (GeV)")
plt.title("Kaon Mass (p = 1) vs. $a^2$ for Different Ensemble")
plt.legend()
plt.grid()
plt.xlim((0,0.008))
# Show plot
plt.show()
# Load the fit results from the files
fit_results = {}
for kappa_h in kappa_hs:
    for i_conf, conf_name in enumerate(conf_names):
        file_name = dir + conf_name + "tree_fit_results_kappa_h_{}.txt".format(kappa_h)
        with open(file_name, "r") as f:
            lines = f.readlines()
            fit_results[(kappa_h, conf_name)] = {}
            for line in lines:
                if line.startswith("f_k = "):
                    fit_results[(kappa_h, conf_name)]["f_k"] = float(line.split("=")[1].split("+/-")[0])
                    fit_results[(kappa_h, conf_name)]["f_k_err"] = float(line.split("+/-")[1])
                elif line.startswith("m_Psi = "):
                    fit_results[(kappa_h, conf_name)]["m_Psi"] = float(line.split("=")[1].split("+/-")[0])
                    fit_results[(kappa_h, conf_name)]["m_Psi_err"] = float(line.split("+/-")[1])
                elif line.startswith("phi^2 = "):
                    fit_results[(kappa_h, conf_name)]["phi2"] = float(line.split("=")[1].split("+/-")[0])
                    fit_results[(kappa_h, conf_name)]["phi2_err"] = float(line.split("+/-")[1])
                elif line.startswith("phi~ = "):
                    fit_results[(kappa_h, conf_name)]["phi"] = float(line.split("=")[1].split("+/-")[0])
                    fit_results[(kappa_h, conf_name)]["phi_err"] = float(line.split("+/-")[1])


# Define a color map for the configurations
conf_colors = {
    "B451": "tab:blue",
    "B452": "tab:orange",
    "N450": "tab:green",
    "N304": "tab:red",
    "N305": "tab:gray",
}
shift = 2e-4
# Function to adjust color based on kappa_h
def adjust_color(base_color, kappa_h, adjustment_factor=0.3):
    # Convert base color to RGB
    rgb_color = mcolors.to_rgb(base_color)
    
    # Create a slight adjustment to the color based on kappa_h
    adjustment = (kappa_hs.index(kappa_h) / len(kappa_hs)) * adjustment_factor
    
    # Adjust the RGB values
    adjusted_rgb = [min(1, max(0, c + adjustment)) for c in rgb_color]
    
    return adjusted_rgb

# Create figure
plt.figure(figsize=(8, 5))

# Loop through kappa_h values and configurations
for conf_name in conf_names:
    for kappa_h in kappa_hs:
        key = (kappa_h, conf_name)
        if key not in fit_results:
            continue  # Skip missing data

        # Extract values
        m_Psi = fit_results[key]["m_Psi"]
        m_Psi_err = fit_results[key]["m_Psi_err"] 
        a = List[conf_names.index(conf_name)][6]  # Extract lattice spacing
        a_m_Psi_sq = (m_Psi) ** 2  # Compute (a m_Psi)^2
        a_m_Psi_sq_err = m_Psi * m_Psi_err
        m_Psi_inv = 1 / m_Psi  * a / fminv_to_GEV# Compute m_Psi^{-1}
        m_Psi_inv_err = m_Psi_err / (m_Psi**2)  * a / fminv_to_GEV# Error propagation
        base_color = conf_colors[conf_name]  # Base color for configuration
        color = adjust_color(base_color, kappa_h)
        # Plot
        plt.errorbar(a_m_Psi_sq, m_Psi_inv, xerr=a_m_Psi_sq_err, yerr=m_Psi_inv_err, fmt="o",
                     label=f"{conf_name}, κ_h={kappa_h}",color=color, capsize=5)

# Labels and formatting
plt.xlabel("${(am_{\Psi})}^2$")
plt.ylabel("${m_{\Psi}}^{-1}$ (${GeV}^{-1}$)")
plt.title("${m_{\Psi}}^{-1}$ vs. ${(am_{\Psi})}^2$ for Different Ensemble")
plt.legend()
plt.grid()
# Show plot
plt.show()

# Create figure
plt.figure(figsize=(8, 5))

# Loop through kappa_h values and configurations
for conf_name in conf_names:
    for kappa_h in kappa_hs:
        key = (kappa_h, conf_name)
        if key not in fit_results:
            continue  # Skip missing data

        # Extract values
        a = List[conf_names.index(conf_name)][6]  # Extract lattice spacing
        a_squared = a ** 2  # Compute a^2
        f_k = fit_results[key]["f_k"] /a * fminv_to_GEV
        f_k_err = fit_results[key]["f_k_err"] /a * fminv_to_GEV
        m_Psi = fit_results[key]["m_Psi"]  /a * fminv_to_GEV # Extract m_Psi for shifting
        
        
        # Shift x-axis (a^2) by subtracting m_Psi
        shifted_a_squared = a_squared - shift * m_Psi  # Subtract m_Psi from a^2

        # Get base color for configuration
        base_color = conf_colors[conf_name]
        
        # Adjust color based on kappa_h
        adjusted_color = adjust_color(base_color, kappa_h)

        # Plot f_k vs shifted a^2 (x-axis)
        plt.errorbar(shifted_a_squared, f_k, yerr=f_k_err, fmt="o",
                     label=f"{conf_name}, κ_h={kappa_h}", color=adjusted_color, capsize=5)

        # Add m_Psi value as annotation inside the plot for each point
        plt.text(shifted_a_squared, f_k, "$m_{\Psi}$:" + f"{m_Psi:.3f}", color=adjusted_color, fontsize=8,
                 ha='right', va='bottom')  # Adjust position for better readability

# Labels and formatting
plt.xlabel(r"$a^2$ (${fm}^2$)")
plt.ylabel(r"$f_k$ (GeV)")
plt.title(r"$f_k$ vs. $(a^2)$ for Different Ensemble")
plt.legend()
plt.grid()
plt.xlim((0,0.008))
# Show plot
plt.show()

# Create figure
plt.figure(figsize=(8, 5))

# Loop through kappa_h values and configurations
for conf_name in conf_names:
    for kappa_h in kappa_hs:
        key = (kappa_h, conf_name)
        if key not in fit_results:
            continue  # Skip missing data

        # Extract values
        phi = fit_results[key]["phi"]
        phi_err = fit_results[key]["phi_err"]
        m_Psi = fit_results[key]["m_Psi"] /a * fminv_to_GEV  # Extract m_Psi for shifting
        a = List[conf_names.index(conf_name)][6]  # Extract lattice spacing
        a_squared = a ** 2  # Compute a^2
        
        # Shift x-axis (a^2) by subtracting m_Psi
        shifted_a_squared = a_squared - shift * m_Psi  # Subtract m_Psi from a^2

        # Get base color for configuration
        base_color = conf_colors[conf_name]
        
        # Adjust color based on kappa_h
        adjusted_color = adjust_color(base_color, kappa_h)

        # Plot f_k vs shifted a^2 (x-axis)
        plt.errorbar(shifted_a_squared, phi, yerr=phi_err, fmt="o",
                     label=f"{conf_name}, κ_h={kappa_h}", color=adjusted_color, capsize=5)

        # Add m_Psi value as annotation inside the plot for each point
        plt.text(shifted_a_squared, phi, "$m_{\Psi}$:" + f"{m_Psi:.3f}", color=adjusted_color, fontsize=8,
                 ha='right', va='bottom')  # Adjust position for better readability

# Labels and formatting
plt.xlabel(r"$a^2$ (${fm}^2$)")
plt.ylabel(r"$\phi$")
plt.title(r"$\phi$ vs. $(a^2)$ for Different Ensemble")
plt.legend()
plt.grid()
plt.xlim((0,0.008))
# Show plot
plt.show()

plt.figure(figsize=(8, 5))
# Loop through kappa_h values and configurations
for conf_name in conf_names:
    for kappa_h in kappa_hs:
        key = (kappa_h, conf_name)
        if key not in fit_results:
            continue  # Skip missing data

        # Extract values
        phi = fit_results[key]["phi2"]
        phi_err = fit_results[key]["phi2_err"]
        m_Psi = fit_results[key]["m_Psi"] /a * fminv_to_GEV  # Extract m_Psi for shifting
        a = List[conf_names.index(conf_name)][6]  # Extract lattice spacing
        a_squared = a ** 2  # Compute a^2
        
        # Shift x-axis (a^2) by subtracting m_Psi
        shifted_a_squared = a_squared - shift * m_Psi  # Subtract m_Psi from a^2

        # Get base color for configuration
        base_color = conf_colors[conf_name]
        
        # Adjust color based on kappa_h
        adjusted_color = adjust_color(base_color, kappa_h)

        # Plot f_k vs shifted a^2 (x-axis)
        plt.errorbar(shifted_a_squared, phi, yerr=phi_err, fmt="o",
                     label=f"{conf_name}, κ_h={kappa_h}", color=adjusted_color, capsize=5)

        # Add m_Psi value as annotation inside the plot for each point
        plt.text(shifted_a_squared, phi, "$m_{\Psi}$:" + f"{m_Psi:.3f}", color=adjusted_color, fontsize=8,
                 ha='right', va='bottom')  # Adjust position for better readability

# Labels and formatting
plt.xlabel(r"$a^2$ (${fm}^2$)")
plt.ylabel(r"${\phi}^2$")
plt.title(r"${\phi}^2$ vs. $(a^2)$ for Different Ensemble")
plt.legend()
plt.grid()
plt.xlim((0,0.008))
# Show plot
plt.show()

# Create figure
plt.figure(figsize=(8, 5))

# Loop through kappa_h values and configurations
for conf_name in conf_names:
    for kappa_h in kappa_hs:
        key = (kappa_h, conf_name)
        if key not in fit_results:
            continue  # Skip missing data

        # Extract values
        a = List[conf_names.index(conf_name)][6]  # Extract lattice spacing
        a_squared = a ** 2  # Compute a^2
        f_k = fit_results[key]["f_k"] /a * fminv_to_GEV
        f_k_err = fit_results[key]["f_k_err"] /a * fminv_to_GEV
        m_Psi = fit_results[key]["m_Psi"]  /a * fminv_to_GEV # Extract m_Psi for shifting
        m_Psi_err = fit_results[key]["m_Psi_err"]  /a * fminv_to_GEV 
        # Get base color for configuration
        base_color = conf_colors[conf_name]
        
        # Adjust color based on kappa_h
        adjusted_color = adjust_color(base_color, kappa_h)

        # Plot f_k vs shifted a^2 (x-axis)
        plt.errorbar(m_Psi, f_k, xerr=m_Psi_err, yerr=f_k_err, fmt="o",
                     label=f"{conf_name}, κ_h={kappa_h}", color=adjusted_color, capsize=5)

# Labels and formatting
plt.xlabel(r"$m_{\Psi}$ (GeV)")
plt.ylabel(r"$f_k$ (GeV)")
plt.title(r"$f_k$ vs. $m_{\Psi}$ for Different Ensemble")
plt.legend()
plt.grid()
# Show plot
plt.show()

# Create figure
plt.figure(figsize=(8, 5))

# Loop through kappa_h values and configurations
for conf_name in conf_names:
    for kappa_h in kappa_hs:
        key = (kappa_h, conf_name)
        if key not in fit_results:
            continue  # Skip missing data

        # Extract values
        phi = fit_results[key]["phi"]
        phi_err = fit_results[key]["phi_err"]
        m_Psi = fit_results[key]["m_Psi"]  /a * fminv_to_GEV # Extract m_Psi for shifting
        m_Psi_err = fit_results[key]["m_Psi_err"]  /a * fminv_to_GEV 
        # Get base color for configuration
        base_color = conf_colors[conf_name]
        
        # Adjust color based on kappa_h
        adjusted_color = adjust_color(base_color, kappa_h)

        # Plot f_k vs shifted a^2 (x-axis)
        plt.errorbar(m_Psi, phi, xerr=m_Psi_err, yerr=phi_err, fmt="o",
                     label=f"{conf_name}, κ_h={kappa_h}", color=adjusted_color, capsize=5)

# Labels and formatting
plt.xlabel(r"$m_{\Psi}$ (GeV)")
plt.ylabel(r"$\phi$")
plt.title(r"$\phi$ vs. $m_{\Psi}$ for Different Ensemble")
plt.legend()
plt.grid()
# Show plot
plt.show()

# Create figure
plt.figure(figsize=(8, 5))

# Loop through kappa_h values and configurations
for conf_name in conf_names:
    for kappa_h in kappa_hs:
        key = (kappa_h, conf_name)
        if key not in fit_results:
            continue  # Skip missing data

        # Extract values
        phi = fit_results[key]["phi2"]
        phi_err = fit_results[key]["phi2_err"]
        m_Psi = fit_results[key]["m_Psi"]  /a * fminv_to_GEV # Extract m_Psi for shifting
        m_Psi_err = fit_results[key]["m_Psi_err"]  /a * fminv_to_GEV 
        # Get base color for configuration
        base_color = conf_colors[conf_name]
        
        # Adjust color based on kappa_h
        adjusted_color = adjust_color(base_color, kappa_h)

        # Plot f_k vs shifted a^2 (x-axis)
        plt.errorbar(m_Psi, phi, xerr=m_Psi_err, yerr=phi_err, fmt="o",
                     label=f"{conf_name}, κ_h={kappa_h}", color=adjusted_color, capsize=5)

# Labels and formatting
plt.xlabel(r"$m_{\Psi}$ (GeV)")
plt.ylabel(r"${\phi}^2$")
plt.title(r"${\phi}^2$ vs. $m_{\Psi}$ for Different Ensemble")
plt.legend()
plt.grid()
# Show plot
plt.show()
