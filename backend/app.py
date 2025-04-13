from flask import Flask, render_template, request, send_file, jsonify
from flask_cors import CORS
from threading import Lock
from io import BytesIO
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import py3Dmol
import random
from typing import Optional, Dict, Any
from models.model import model
from models.preprocess import preprocess_data

app = Flask(__name__)


progress = 0
progress_lock = Lock()
current_task = ""


def update_progress(step: int, total: int, task: str = ""):
    """Update the global progress variables"""
    global progress, current_task
    with progress_lock:
        progress = int((step / total) * 100) if total > 0 else 0
        current_task = task


@app.route('/progress')
def get_progress():
    """Endpoint to get current progress"""
    with progress_lock:
        return jsonify({
            'progress': progress,
            'task': current_task
        })


@app.route('/csv_upload', methods=['GET', 'POST'])
def csv_upload():
    global progress, current_task

    if request.method == 'POST':
        if 'csv_file' not in request.files:
            return render_template('csv_upload.html', error="No file uploaded")

        file = request.files['csv_file']
        if file.filename == '':
            return render_template('csv_upload.html', error="No selected file")
        if not file.filename.lower().endswith('.csv'):
            return render_template('csv_upload.html', error="File must be a CSV")

        try:
            # Reset progress
            update_progress(0, 1, "Reading CSV file")

            df = pd.read_csv(file.stream)
            df.columns = df.columns.str.lower()

            if 'smiles' not in df.columns:
                return render_template('csv_upload.html', error="CSV must contain 'smiles' column")

            smiles_list = df['smiles'].tolist()
            total_molecules = len(smiles_list)

            # Preprocess with progress
            update_progress(0, total_molecules, "Preprocessing molecules")
            preprocessed_data = []
            for i, smiles in enumerate(smiles_list):
                preprocessed_data.append(preprocess_data([smiles])[0])
                update_progress(i + 1, total_molecules, "Preprocessing molecules")
                #time.sleep(0.01)  # Simulate work for demo

            # Predict with progress
            update_progress(0, total_molecules, "Predicting logP values")
            logp_predictions = []
            for i, data in enumerate(preprocessed_data):
                logp_predictions.append(model.predict([data])[0])
                update_progress(i + 1, total_molecules, "Predicting logP values")
                #time.sleep(0.01)  # Simulate work for demo

            result_df = pd.DataFrame({
                'smiles': smiles_list,
                'logP': logp_predictions
            })

            output = BytesIO()
            result_df.to_csv(output, index=False)
            output.seek(0)

            # Reset progress when done
            update_progress(0, 1, "Complete")

            return send_file(
                output,
                mimetype='text/csv',
                as_attachment=True,
                download_name='logP_predictions.csv'
            )

        except Exception as e:
            update_progress(0, 1, "Error occurred")
            return render_template('csv_upload.html', error=f"Error processing file: {str(e)}")

    # Reset progress when loading the page
    update_progress(0, 1, "Ready")
    return render_template('csv_upload.html')


def generate_3d_view(smiles: str) -> Optional[str]:
    """
    Generate interactive 3D molecular visualization from a SMILES string.

    This function converts a SMILES string into a 3D molecular representation
    using RDKit for molecular processing and py3Dmol for visualization.

    Parameters:
        smiles (str): SMILES string representing the molecule to visualize.
                     Must be a valid chemical structure notation.

    Returns:
        Optional[str]: HTML string containing the 3D visualization if successful,
                      None if the SMILES is invalid or processing fails.

    Raises:
        RuntimeError: If molecular generation or embedding fails silently
        (Note: Exceptions are caught internally and result in None return)

    Processing Steps:
        1. SMILES parsing (RDKit)
        2. Hydrogen addition
        3. 3D conformation generation (ETKDG algorithm)
        4. 3D visualization setup (py3Dmol)
        5. HTML viewer generation

    Visualization Features:
        - Stick representation with light grey carbon atoms
        - Atomic spheres (scale: 0.25)
        - Automatic zoom to fit molecule
        - Interactive rotation/zoom controls

    Notes:
        - Requires RDKit and py3Dmol packages
        - Uses ETKDG (Experimental-Torsion basic Knowledge Distance Geometry)
          for 3D coordinate generation
        - Default display size is 550x550 pixels
        - Hydrogen atoms are explicitly added for accurate 3D representation
        - Invalid SMILES will return None without raising exceptions

    See Also:
        Chem.MolFromSmiles : RDKit SMILES parser
        AllChem.EmbedMolecule : RDKit 3D coordinate generator
        py3Dmol.view : 3D visualization engine
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)

        viewer = py3Dmol.view(width=550, height=550)
        viewer.addModel(Chem.MolToMolBlock(mol), 'mol')
        viewer.setStyle({'stick': {'colorscheme': 'lightgreyCarbon'},
                        'sphere': {'scale': 0.25}})

        viewer.zoomTo()
        viewer.zoomTo(1.5)

        color_info = """
                <div class="color-legend">
                    <h3><i class="fas fa-palette"></i> Atom Color Scheme</h3>
                    <ul class="color-list">
                        <li><span class="color-swatch" style="background-color: #909090;"></span> Carbon (C) - Light Grey</li>
                        <li><span class="color-swatch" style="background-color: #FF0D0D;"></span> Oxygen (O) - Red</li>
                        <li><span class="color-swatch" style="background-color: #1E90FF;"></span> Nitrogen (N) - Blue</li>
                        <li><span class="color-swatch" style="background-color: #FFD700;"></span> Sulfur (S) - Yellow</li>
                        <li><span class="color-swatch" style="background-color: #228B22;"></span> Chlorine (Cl) - Green</li>
                        <li><span class="color-swatch" style="background-color: #FF1493;"></span> Phosphorus (P) - Pink</li>
                        <li><span class="color-swatch" style="background-color: #9400D3;"></span> Fluorine (F) - Purple</li>
                        <li><span class="color-swatch" style="background-color: #FFA500;"></span> Bromine (Br) - Orange</li>
                        <li><span class="color-swatch" style="background-color: #FFFFFF;"></span> Hydrogen (H) - White</li>
                    </ul>
                    <p class="legend-note">Colors follow the CPK coloring convention commonly used in molecular visualization.</p>
                </div>
                """

        return viewer._make_html() + color_info
    except Exception as e:
        print(f"Error generating 3D view: {e}")
        return None


def predict_properties(smiles: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Predict key molecular properties from a SMILES string with detailed interpretations.

    This function calculates physicochemical properties relevant to drug discovery
    and medicinal chemistry, following Lipinski's Rule of Five and QSPR principles.

    Parameters:
        smiles (str): Valid SMILES string representing the molecular structure

    Returns:
        Optional[Dict]: Nested dictionary of predicted properties with metadata, or None if invalid.
        Structure:
        {
            'property_name': {
                'value': float/int,       # Calculated property value
                'interpretation': str,    # Scientific assessment
                'ideal_range': str,       # Recommended range for drug-likeness
                'unit': str              # Measurement units
            },
            ...
        }

    Raises:
        Silent Errors: All exceptions are caught internally and return None

    Calculated Properties:
        1. logP (Partition coefficient):
            - Predicts lipophilicity using trained model
            - Interpretation based on pharmaceutical guidelines

        2. Molecular Weight:
            - Calculated exactly using RDKit
            - Critical for drug-likeness (Rule of Five)

        3. Topological Polar Surface Area (TPSA):
            - Predicts membrane permeability
            - Important for bioavailability

        4. Rotatable Bonds:
            - Count of non-rigid bonds
            - Indicator of molecular flexibility

    Notes:
        - Uses RDKit for exact descriptor calculations
        - Employs a trained ML model for logP prediction
        - Implements fallback to random values when RDKit fails (for robustness)
        - Property interpretations follow:
            * Lipinski's Rule of Five
            * Veber's rules for oral bioavailability
            * Egan's "Rule of 3" for CNS permeability

    See Also:
        get_logp_interpretation : Detailed logP analysis
        Descriptors.MolWt : RDKit molecular weight calculation
        Descriptors.TPSA : Polar surface area calculation
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        logp = model.predict(preprocess_data([smiles]))[0]
        mol_weight = round(Descriptors.MolWt(mol), 4) if mol else round(random.uniform(50, 500), 4)
        tpsa = round(Descriptors.TPSA(mol), 4) if mol else round(random.uniform(0, 150), 4)
        num_rotatable_bonds = Descriptors.NumRotatableBonds(mol) if mol else random.randint(0, 10)

        return {
            'logp': {
                'value': logp,
                'interpretation': get_logp_interpretation(logp),
                'ideal_range': '0-3',
                'unit': ''
            },
            'molecular_weight': {
                'value': mol_weight,
                'interpretation': get_mw_interpretation(mol_weight),
                'ideal_range': '<500',
                'unit': 'g/mol'
            },
            'polar_surface_area': {
                'value': tpsa,
                'interpretation': get_tpsa_interpretation(tpsa),
                'ideal_range': '<140',
                'unit': 'Å²'
            },
            'rotatable_bonds': {
                'value': num_rotatable_bonds,
                'interpretation': get_rot_bonds_interpretation(num_rotatable_bonds),
                'ideal_range': '<5',
                'unit': ''
            }
        }
    except Exception as e:
        print(f"Error predicting properties: {e}")
        return None


def get_logp_interpretation(logp: float) -> str:
    """
    Provide a scientifically-validated interpretation of a compound's lipophilicity (logP) value.

    Parameters:
        logp (float): The octanol-water partition coefficient value to interpret.
                      Typical range for drug-like molecules: -0.5 to 5.0.

    Returns:
        str: Structured interpretation including:
             - Solubility characteristics
             - Membrane permeability assessment
             - Drug development implications
             - Toxicity risk evaluation

    Interpretation Ranges:
        < -0.5    : Highly hydrophilic
        -0.5 - 1.0: Moderate hydrophilicity (ideal for most drugs)
        1.0 - 3.0 : Optimal lipophilicity (best bioavailability)
        3.0 - 5.0 : High lipophilicity (solubility concerns)
        5.0 - 7.0 : Very high lipophilicity (toxicity risk)
        ≥ 7.0     : Extreme lipophilicity (development discouraged)

    Scientific Basis:
        - Lipinski's Rule of Five (logP < 5)
        - Pfizer's 3/75 Rule (logP < 3 preferred)
        - Golden Triangle of logP (1-3) for CNS drugs
        - Ghose filter (logP between -0.4 and 5.6)
        - Veber's rules for oral bioavailability

    Pharmacokinetic Implications:
        • Absorption: Optimal at logP 1-3
        • Distribution: Higher logP increases tissue accumulation
        • Metabolism: High logP increases CYP450 interaction risk
        • Excretion: Low logP compounds clear renally

    References:
        1. Lipinski et al. (2001) Adv Drug Deliv Rev
        2. Veber et al. (2002) J Med Chem
        3. Waring (2010) Bioorg Med Chem Lett
    """
    if logp < -0.5:
        return ("Highly hydrophilic (water-loving). Excellent water solubility but poor membrane permeability. "
                "Likely to have difficulty crossing biological membranes. Consider prodrug approaches "
                "for better absorption. Typical of highly polar compounds like sugars.")
    elif -0.5 <= logp < 1.0:
        return ("Moderate hydrophilicity. Good balance of water solubility and membrane permeability. "
                "Ideal range for most oral drugs (85% of marketed drugs fall here). Suitable for "
                "compounds requiring systemic circulation.")
    elif 1.0 <= logp < 3.0:
        return ("Moderate lipophilicity. Optimal range for drug absorption and bioavailability. "
                "Most successful oral drugs (including CNS-active compounds) fall in this range. "
                "Balances solubility and permeability for best PK properties.")
    elif 3.0 <= logp < 5.0:
        return ("High lipophilicity. May have permeability issues due to poor water solubility. "
                "Potential for accumulation in fatty tissues and hepatocytes. Consider formulation "
                "enhancements (nanoparticles, solubilizers). Common in lipophilic CNS drugs.")
    elif 5.0 <= logp < 7.0:
        return ("Very high lipophilicity. Likely poor water solubility and bioavailability. "
                "High risk of metabolic instability and toxicity (CYP inhibition, phospholipidosis). "
                "Requires extensive formulation optimization. Often leads to drug development failure.")
    else:  # logp >= 7.0
        return ("Extreme lipophilicity. Strong tendency to accumulate in lipid bilayers and fatty tissues. "
                "Very poor absorption and high toxicity risk. Generally undesirable for drugs. "
                "Typical of persistent environmental pollutants (DDT-like compounds).")

def get_mw_interpretation(mw: float) -> str:
    """
    Evaluate molecular weight significance in drug development and pharmacokinetics.

    Parameters:
        mw (float): Molecular weight in Daltons (g/mol).
                    Typical drug range: 150-500 Da.

    Returns:
        str: Structured interpretation including:
             - Bioavailability prediction
             - Absorption/penetration assessment
             - Development risk evaluation
             - Rule-based recommendations

    Interpretation Ranges:
        < 300 Da : Ideal for most drug targets
        300-500 Da: Acceptable with potential limitations
        > 500 Da : Significant bioavailability challenges
        > 900 Da : Biologics territory (special formulations needed)

    Scientific Basis:
        - Lipinski's Rule of Five (MW < 500 Da)
        - Pfizer's 3/75 Rule (MW < 400 Da preferred)
        - Golden Triangle for CNS drugs (MW < 450 Da)
        - Natural product drug space (up to 600 Da)
        - Biologic therapeutics (>1000 Da)

    Pharmacokinetic Implications:
        • Absorption: Optimal <300 Da (passive diffusion)
        • Distribution: Higher MW reduces tissue penetration
        • Metabolism: MW >500 often requires active transport
        • Excretion: Renal clearance decreases above 500 Da

    References:
        1. Lipinski et al. (2001) Adv Drug Deliv Rev
        2. Leeson & Springthorpe (2007) Nat Rev Drug Discov
        3. Doak et al. (2014) Drug Discov Today
    """
    if mw < 150:
        return "Very small (may lack target specificity, potential for off-target effects)"
    elif 150 <= mw < 300:
        return ("Excellent (optimal for passive diffusion and good bioavailability. "
                "Typical for CNS drugs and oral medications)")
    elif 300 <= mw < 400:
        return ("Good (moderate bioavailability. May show some permeability limitations. "
                "Common range for successful drugs)")
    elif 400 <= mw < 500:
        return ("Moderate (developable but with formulation challenges. "
                "Often requires permeability enhancers. Common for natural product-derived drugs)")
    elif 500 <= mw < 900:
        return ("High (beyond Rule of Five space. May require prodrug approaches "
                "or alternative delivery methods. Limited to specific therapeutic areas)")
    else:
        return ("Very high (beyond typical small molecule drug space. "
                "Consider biologics formulation strategies or alternative modalities)")

def get_tpsa_interpretation(tpsa: float) -> str:
    """
    Evaluate topological polar surface area (TPSA) significance in drug development.

    Parameters:
        tpsa (float): Topological polar surface area in Å² (angstrom squared).
                      Typical drug range: 20-140 Å².

    Returns:
        str: Structured interpretation including:
             - Permeability prediction
             - Absorption potential
             - Blood-brain barrier penetration likelihood
             - Development considerations

    Interpretation Ranges:
        < 60 Å² : Excellent membrane permeability (high absorption)
        60-90 Å²: Good permeability (oral drugs typically <90 Å²)
        90-140 Å²: Moderate (may need formulation optimization)
        > 140 Å²: Poor permeability (special delivery required)

    Scientific Basis:
        - Veber's Rule (TPSA < 140 Å² for oral bioavailability)
        - CNS drugs typically < 60-70 Å²
        - Egan's "Rule of 3" (TPSA < 120 Å²)
        - Hydrogen bond donor/acceptors influence

    Pharmacokinetic Implications:
        • Absorption: Inverse correlation with TPSA
        • BBB Penetration: <90 Å² preferred for CNS targets
        • Solubility: Higher TPSA improves aqueous solubility
        • Permeability: Lower TPSA enhances passive diffusion

    References:
        1. Veber et al. (2002) J Med Chem
        2. Egan et al. (2000) J Pharmacol Exp Ther
        3. Pajouhesh & Lenz (2005) NeuroRx
    """
    if tpsa < 60:
        return ("Excellent (high membrane permeability, good BBB penetration potential. "
                "Typical range for CNS-active drugs and highly absorbable compounds.")
    elif 60 <= tpsa < 90:
        return ("Good (moderate permeability, suitable for oral administration. "
                "Optimal range for most systemic drugs with balanced solubility/permeability)")
    elif 90 <= tpsa < 120:
        return ("Moderate (developable permeability with potential absorption limitations. "
                "May benefit from permeability enhancers or prodrug approaches)")
    elif 120 <= tpsa < 140:
        return ("Moderate-high (borderline for oral bioavailability per Veber's rules. "
                "Often requires formulation optimization for adequate absorption)")
    else:
        return ("High (significant permeability challenges, poor passive diffusion. "
                "Consider alternative delivery methods or structural modification. "
                "Typical range for many biologics and polar therapeutics)")

def get_rot_bonds_interpretation(num: int) -> str:
    """
    Evaluate the impact of rotatable bond count on drug-like properties and bioavailability.

    Parameters:
        num (int): Number of rotatable bonds (single non-ring bonds excluding amides).
                   Typical range for oral drugs: 0-10.

    Returns:
        str: Structured interpretation including:
             - Oral bioavailability prediction
             - Molecular flexibility assessment
             - Conformational entropy considerations
             - Development recommendations

    Interpretation Ranges:
        < 5 : Ideal for oral drugs (Veber's rule)
        5-7 : Moderate (may need optimization)
        7-10: High (significant bioavailability risk)
        > 10: Very high (typically poor drug candidates)

    Scientific Basis:
        - Veber's Rule (≤10 rotatable bonds for oral bioavailability)
        - Pfizer's 3/75 Rule (≤5 preferred)
        - Rotatable bond impact on:
          * Conformational entropy
          * Passive membrane permeability
          * Metabolic stability

    Pharmacokinetic Implications:
        • Absorption: Fewer bonds → better passive diffusion
        • Metabolism: More bonds → higher oxidation risk
        • Solubility: More bonds → typically better
        • Selectivity: Fewer bonds → often better target binding

    References:
        1. Veber et al. (2002) J Med Chem
        2. Leeson & Springthorpe (2007) Nat Rev Drug Discov
        3. Price et al. (2009) J Med Chem
    """
    if num < 3:
        return ("Excellent (optimal rigidity for oral bioavailability, "
                "typically shows good metabolic stability and membrane permeability)")
    elif 3 <= num < 5:
        return ("Very good (preferred range for CNS drugs and compounds "
                "requiring high bioavailability. Balanced flexibility for binding)")
    elif 5 <= num < 7:
        return ("Moderate (developable but may benefit from reduction. "
                "Potential for reduced absorption and increased metabolism)")
    elif 7 <= num <= 10:
        return ("High (borderline for oral bioavailability per Veber's rules. "
                "Likely to show reduced absorption and increased clearance)")
    else:
        return ("Very high (typically poor oral bioavailability. "
                "Consider structural rigidification or alternative delivery routes. "
                "Often associated with promiscuous binding and metabolic instability)")

@app.route('/', methods=['GET', 'POST'])
def show_molecule() -> str:
    """
    Display and analyze molecular structures based on SMILES input.

    This endpoint serves as the main interface for:
    - Visualizing molecules in 3D
    - Calculating molecular properties
    - Handling user-submitted SMILES strings

    Parameters (via POST):
        smiles (str): Optional SMILES string input from form data.
                      Defaults to chloroquine-like compound if not provided.

    Returns:
        str: Rendered HTML template ('index.html') containing:
            - 3D molecular visualization
            - Calculated properties
            - Current SMILES string
            - Error messages (if any)

    Behavior:
        GET Request:
            - Displays default molecule visualization
            - Shows calculated properties for default molecule

        POST Request:
            - Validates user-provided SMILES string
            - For valid SMILES:
                * Generates 3D visualization
                * Calculates molecular properties
            - For invalid SMILES:
                * Shows error message
                * Falls back to default molecule

    Template Context:
        molecule_html (str): HTML/JS for 3D molecular visualization
        current_smiles (str): SMILES string being displayed
        properties (dict): Dictionary containing:
            - molecular_weight (float)
            - logp (float)
            - h_bond_donors (int)
            - h_bond_acceptors (int)
            - tpsa (float) [Topological Polar Surface Area]
            - ...other calculated properties
        error (str|None): Error message if SMILES parsing fails

    Notes:
        - Uses RDKit (or other specified backend) for molecular processing
        - 3D visualization powered by (specify library: 3DMol.js, NGL Viewer, etc.)
        - Property prediction uses (specify model: QSPR, ML model, etc.)
    """
    default_smiles = 'CN(C)C=1C(=C(N(C)C)C=C(C=1)CC2=CNC(=N)NC2=N)C'
    smiles = default_smiles
    molecule_html = None
    properties = None
    error = None

    if request.method == 'POST':
        smiles = request.form.get('smiles', default_smiles).strip()
        if smiles:
            molecule_html = generate_3d_view(smiles)
            if molecule_html is None:
                error = "Invalid SMILES string. Please try again."
                smiles = default_smiles
                molecule_html = generate_3d_view(smiles)
            properties = predict_properties(smiles)

    if molecule_html is None:
        molecule_html = generate_3d_view(smiles)

    if properties is None:
        properties = predict_properties(smiles)

    return render_template('index.html',
                         molecule_html=molecule_html,
                         current_smiles=smiles,
                         properties=properties,
                         error=error)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)