import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from icecream import ic
from tqdm import tqdm
import STOM_higgs_tools
from STOM_higgs_tools import get_B_chi, get_SB_expectation 
from scipy.stats import chi2

def exponential_distribution(x: np.ndarray, A: float, lamb: float) -> np.ndarray:
    #Exponential distribution function for background events.

    return A * np.exp(-x/lamb)


def gaussian_distribution(x: np.ndarray, B: float, mu: float, sigma: float) -> np.ndarray:
    #Normalized Gaussian distribution function for signal events. 
    return B/np.sqrt(2*np.pi*sigma**2) * np.exp(-(x-mu)**2 / (2*sigma**2))


def combined_distribution(x: np.ndarray, *params) -> np.ndarray:
    #Combined distribution of exponential background and Gaussian signal.
    A, lamb, B, mu, sigma = params
    return exponential_distribution(x, A, lamb) + gaussian_distribution(x, B, mu, sigma)


MASS_RANGE = (104, 155)
NUM_BINS = 30
# Estimate background parameters this is perceived between the two vertical lines in the histogram
SIGNAL_REGION = (120, 130)

def main():
    # Generate and prepare data
    amplitude_values = np.array(STOM_higgs_tools.generate_data())
    
    # Define histogram parameters
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # plt.hist returns bin_heights, bin_edges, patches
    # where bin_heights is the number of events in each bin
    # bin_edges is the coordinates of the bin edges
    bin_heights, bin_edges, _ = plt.hist(
        amplitude_values, 
        bins=NUM_BINS, 
        alpha=0, 
        color='blue', 
        edgecolor='black', 
        range=MASS_RANGE
    )
    
    # Calculate bin properties
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    uncertainties = np.sqrt(bin_heights)  # Poisson uncertainty

    bin_width = bin_edges[1] - bin_edges[0]
    x_uncertainties = bin_width / 2
    
    background_values = np.concatenate((
        amplitude_values[amplitude_values < SIGNAL_REGION[0]],
        amplitude_values[amplitude_values > SIGNAL_REGION[1]]
    ))
    bin_heights_signal = bin_heights[
        (bin_centers >= SIGNAL_REGION[0]) & 
        (bin_centers <= SIGNAL_REGION[1])
    ]
    bin_centers_signal = bin_centers[
        (bin_centers >= SIGNAL_REGION[0]) & 
        (bin_centers <= SIGNAL_REGION[1])
    ]
    # finding the index of the peak of the bin_heights_signal and using that as gaussian mean
    peak_index,_ = find_peaks(bin_heights_signal)
    mu_estimate = bin_centers_signal[peak_index[0]]
    # take sigma as the standard deviation of the histogram within the signal range
    sigma_estimate = np.std(bin_centers_signal)

    # Estimate lambda for exponential background
    lamb_estimate_max_likelihood = np.sum(background_values) / len(background_values)
    
    # Calculate amplitude for exponential background
    # by taking an integration of the background distribution over the mass range
    # and dividing by the area under the histogram
    area_under_histogram = np.sum(bin_heights) * bin_width
    A_estimate_max_likelihood = area_under_histogram / (
        lamb_estimate_max_likelihood * 
        (np.exp(-MASS_RANGE[0]/lamb_estimate_max_likelihood) - 
         np.exp(-MASS_RANGE[1]/lamb_estimate_max_likelihood))
    )
    # B_estimate has been obtained by taking the height of the peak of the gaussian shape in the histogram
    B_estimate = np.max(bin_heights)


    # Initial parameter guesses for curve fitting
    initial_params = [
        A_estimate_max_likelihood,    # A
        lamb_estimate_max_likelihood,  # lambda
        B_estimate,                 # B
        mu_estimate,              # mu
        sigma_estimate                  # sigma
    ]
    
    # Perform curve fitting
    fitted_params, _ = curve_fit(
        combined_distribution, 
        bin_centers, 
        bin_heights, 
        p0=initial_params
    )
    
    # Plot results
    x_fine = np.linspace(*MASS_RANGE, 1000)

    

    # Plot combined distribution
    ax.plot(
        x_fine, 
        combined_distribution(x_fine, *fitted_params),
        label='Combined Distribution'
    )
    
    # Plot background distribution
    ax.plot(
        x_fine,
        exponential_distribution(x_fine, *fitted_params[:2]),
        label='Exponential Distribution'
    )

    ax.plot(x_fine, exponential_distribution(x_fine, A_estimate_max_likelihood, lamb_estimate_max_likelihood), 'g', label='Guess from max likelihood')
    
    # Plot error bars
    ax.errorbar(
        bin_centers, 
        bin_heights, 
        yerr=uncertainties, 
        xerr=x_uncertainties, 
        fmt='k.', 
        capsize=3, 
        alpha=0.5
    )
    
    # Customize plot
    ax.set_xlabel('Mass (GeV)')
    ax.set_ylabel('Number of Events')
    ax.set_title('Distribution of Signal and Background Events')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig('hist.png')
    # Save and display plot
    # plt.savefig('hist.png')
    # plt.show()


########################################################
# task3
########################################################
    # function to calculate chi-square for a given bin
    def chi_square(bin_edges, amplitude_values, A, lamb):
        # left_bin_edges = bin_edges[bin_edges < 121]
        left_bin_edges = bin_edges[bin_edges < SIGNAL_REGION[0]]
        right_bin_edges = bin_edges[bin_edges > SIGNAL_REGION[1]]
        # -1 because the bin edges are not included in the bin_heights
        N_para_left = len(left_bin_edges)-1
        N_para_right = len(right_bin_edges)-1
        chi_left = get_B_chi(amplitude_values, (MASS_RANGE[0], left_bin_edges[0]), left_bin_edges, A, lamb)
        chi_right = get_B_chi(amplitude_values, (right_bin_edges[-1], MASS_RANGE[1]), right_bin_edges, A, lamb)
        total_chi = chi_left + chi_right
        total_dof = N_para_left + N_para_right-4
        # reduce chi-square to a single value
        return total_chi/total_dof

    def iterate_chi(A, lamb, n=10):
        lambda_range = np.linspace(lamb*0.99, lamb*1.01, n)
        A_range = np.linspace(A*0.99, A*1.01, n) 
        # Create meshgrid for 3D plotting
        LAMBDA, A = np.meshgrid(A_range, lambda_range)
        chi_square_values = np.zeros_like(LAMBDA) 
        # Calculate chi-square values for each combination
        for i, A_val in enumerate(tqdm(A_range)):
            for j, lambda_val in enumerate(lambda_range):
                chi_square_values[i, j] = chi_square(bin_edges, amplitude_values, 
                                                A_val, lambda_val) 
        # Find the chi-square value closest to 1
        min_diff_to_one = np.argmin(np.abs(chi_square_values - 1))
        # Get the corresponding A and lambda values
        A_idx = min_diff_to_one//len(A_range)
        lambda_idx = min_diff_to_one%len(lambda_range)
        A_min = A_range[A_idx]
        lamb_min = lambda_range[lambda_idx]
        chi_min = chi_square_values[A_idx, lambda_idx]

        #save LAMBDA, A, chi_square_values to a npy file
        np.save('chi_square_values.npy', (LAMBDA, A, chi_square_values))

        # save min values to a file
        np.savetxt('min_values.txt', (A_min, lamb_min, chi_min))
        fig_chi = plt.figure(figsize=(10, 8))
        ax_chi = fig_chi.add_subplot(111, projection='3d')
        ax_chi.plot_surface(LAMBDA, A, chi_square_values, cmap='viridis')
        ax_chi.set_xlabel('Lambda')
        ax_chi.set_ylabel('A')
        ax_chi.set_zlabel('Chi-square')
        ax_chi.set_title('Chi-square for different Lambda and A values') 
        fig_chi.savefig('chi_square_surface.png')
        return A_min, lamb_min, chi_min
    def load_chi_square_values():
        LAMBDA, A, chi_square_values = np.load('chi_square_values.npy')
        return LAMBDA, A, chi_square_values
    def load_min_values():
        A_min, lamb_min, chi_min = np.loadtxt('min_values.txt')
        return A_min, lamb_min, chi_min
    
    # use maximum likelihood as a baseline to do chi-square optimization 
    # A_min, lamb_min, chi_min = iterate_chi(A_estimate_max_likelihood, lamb_estimate_max_likelihood,20)
    LAMBDA, A, chi_square_values = load_chi_square_values()
    A_chi, lamb_chi, chi_min = load_min_values()
    chi_square_calc_max_likelihood = chi_square(bin_edges, amplitude_values, A_estimate_max_likelihood, lamb_estimate_max_likelihood)
    ic(chi_square_calc_max_likelihood)
    # plot these value on the graph
    ax.plot(x_fine, exponential_distribution(x_fine, A_chi, lamb_chi), 'r', label='Guess from Chi-square')
    ax.legend()

    ########################################################
    # task4
    ########################################################
    # A: do a chi square test for the background and signal
    ########################################################
    chi_background_and_signal = get_B_chi(amplitude_values, MASS_RANGE, NUM_BINS, A_chi, lamb_chi)
    ic(chi_background_and_signal)
    
    # Calculate p-value and confidence thresholds
    data_points = len(bin_edges) - 1  # degrees of freedom (number of bins - number of parameters)
    # because of two degree of freedom, we need to subtract 2, (A and lambda)
    dof = data_points - 2
    p_value = 1 - chi2.cdf(chi_background_and_signal, dof)
    
    # Calculate chi-square values for different confidence levels
    chi2_90 = chi2.ppf(0.90, dof)
    chi2_95 = chi2.ppf(0.95, dof)
    chi2_99 = chi2.ppf(0.99, dof)
    
    ic(p_value)
    ic(chi2_90, chi2_95, chi2_99)

    ########################################################
    # B: do a chi square test for the background and signal
    ########################################################
    B_TEST = 700
    MU_TEST = 125
    SIGMA_TEST = 1.5
    test_params = [A_chi, lamb_chi, B_TEST, MU_TEST, SIGMA_TEST]
        
    def get_SB_chi(bin_centers, bin_heights, fitted_params):
        ys_expected = combined_distribution(bin_centers, *fitted_params)
        # Loop over bins - all of them for now. 
        chi = np.sum((bin_heights - ys_expected)**2 / ys_expected)
        return chi 
    
    ax.plot(x_fine, combined_distribution(x_fine, *test_params), 'y', label='Test distribution')
    ax.legend()

    # Calculate chi-square with test parameters
    chi_combined = get_SB_chi(bin_centers, bin_heights, test_params) 
    dof = len(bin_edges) - 5 - 1
    ic(chi_combined)
    ic(dof)
    ic(chi_combined/dof)
    # Calculate p-value and confidence thresholds
    data_points = len(bin_edges) - 1
    dof = data_points - 5  # 5 parameters: A, lambda, B, mu, sigma
    p_value = 1 - chi2.cdf(chi_combined, dof)
    
    # Calculate chi-square values for different confidence levels
    chi2_99 = chi2.ppf(0.99, dof)
    chi2_999 = chi2.ppf(0.999, dof)
    chi2_9999 = chi2.ppf(0.9999, dof)
    chi2_99999 = chi2.ppf(0.99999, dof)
    chi2_999999 = chi2.ppf(0.999999, dof)
    chi2_9999999 = chi2.ppf(0.9999999, dof)
    chi2_9999999 = chi2.ppf(0.9999999, dof)
    
    ic("Chi-square for combined distribution:", chi_combined)
    ic("p-value:", p_value)
    ic("Chi-square thresholds (99%, 99.9%, 99.99%, 99.999%, 99.9999%, 99.99999%):", chi2_99, chi2_999, chi2_9999, chi2_99999, chi2_999999, chi2_9999999)
    ic("Acceptance at 99%:", chi_combined < chi2_99)
    ic("Acceptance at 99.9%:", chi_combined < chi2_999)
    ic("Acceptance at 99.99%:", chi_combined < chi2_9999)
    ic("Acceptance at 99.999%:", chi_combined < chi2_99999)
    ic("Acceptance at 99.9999%:", chi_combined < chi2_999999)
    ic("Acceptance at 99.99999%:", chi_combined < chi2_9999999)

########################################################
# C: find the amplitude corresponding to a p-value of 0.05
########################################################
    def find_amplitude_for_p_value(A_guess, lamb_guess, target_p_value=0.05): 
        # Binary search to find B
        B_min = 0
        B_max = 1000  # Start with a reasonable upper bound
        tolerance = 0.001
        max_iterations = 50
        
        for _ in range(max_iterations):
            B_test = (B_min + B_max) / 2
            test_params = [A_guess, lamb_guess, B_test, MU_TEST, SIGMA_TEST]
            
            # Calculate chi-square and p-value
            chi_combined = get_SB_chi(bin_centers, bin_heights, test_params)
            dof = len(bin_edges) - 5 - 1
            current_p_value = 1 - chi2.cdf(chi_combined, dof)
            
            if abs(current_p_value - target_p_value) < tolerance:
                return B_test
            
            if current_p_value > target_p_value:
                B_max = B_test
            else:
                B_min = B_test
        
        return (B_min + B_max) / 2  # Return the best approximation
    
    # Find the amplitude that corresponds to a p-value of 0.05
    amplitude_for_p_value = find_amplitude_for_p_value(A_chi, lamb_chi, 0.05)
    ic("Signal amplitude B for p-value 0.05:", amplitude_for_p_value)
    
    # Verify the result
    test_params = [A_guess_from_chi, lamb_guess_from_chi, amplitude_for_p_value, 125, 1.5]
    chi_combined = get_SB_chi(bin_centers, bin_heights, test_params)
    dof = len(bin_edges) - 5 - 1
    p_value = 1 - chi2.cdf(chi_combined, dof)
    ic("Verification - p-value:", p_value)
########################################################
# D: plot chi_square for different mu values
########################################################
    # new plot
    fig_mu, ax_mu = plt.subplots()
    def find_chi_for_mu(mu_values):
        chi_values = []
        for mu in mu_values:
            test_params = [A_guess_from_chi, lamb_guess_from_chi, B_TEST, mu, SIGMA_TEST]
            chi_combined = get_SB_chi(bin_centers, bin_heights, test_params)
            chi_values.append(chi_combined)
        return chi_values
    
    mu_values = np.linspace(MU_TEST-10, MU_TEST+10, 100)
    chi_values = find_chi_for_mu(mu_values)
    ax_mu.plot(mu_values, chi_values)
    ax_mu.set_xlabel('Mu')
    ax_mu.set_ylabel('Chi-square')
    ax_mu.set_title('Chi-square for different mu values')
    fig_mu.savefig('chi_square_mu.png')
    plt.show()
        











if __name__ == "__main__":
    main()
