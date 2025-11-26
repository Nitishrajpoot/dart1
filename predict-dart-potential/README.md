# DART Toxicity Predictor

A modern full-stack web application for predicting developmental and reproductive toxicity (DART) of chemical compounds using machine learning models trained on ToxCast data.

## Features

- 🧪 **Single Chemical Analysis** - Predict toxicity for individual compounds using SMILES notation
- 📊 **Batch Processing** - Analyze multiple compounds simultaneously
- 🤖 **AI-Powered** - Ensemble machine learning models (Random Forest, Gradient Boosting, SVM)
- 📈 **Detailed Results** - Probability scores, confidence intervals, and risk assessments
- 💾 **Export Results** - Download batch predictions as CSV files
- 🎨 **Modern UI** - Clean, professional interface built with Next.js and shadcn/ui

## Tech Stack

- **Frontend**: Next.js 16, React 19, TypeScript
- **UI**: shadcn/ui, Tailwind CSS v4
- **Backend**: Next.js API Routes
- **ML Integration**: Ready for Python ML service integration

## Getting Started

### Prerequisites

- Node.js 18+ installed
- npm or yarn package manager

### Installation

1. Clone the repository:
\`\`\`bash
git clone <your-repo-url>
cd dart-predictor
\`\`\`

2. Install dependencies:
\`\`\`bash
npm install
\`\`\`

3. Run the development server:
\`\`\`bash
npm run dev
\`\`\`

4. Open [http://localhost:3000](http://localhost:3000) in your browser

## Deployment

### Deploy to Vercel (Recommended)

1. Push your code to GitHub
2. Import your repository in Vercel
3. Deploy with one click

The app is production-ready and will automatically deploy to Vercel with optimal configuration.

### Environment Variables

Currently, the app uses mock predictions for demonstration. To integrate with a real ML model:

1. Set up your Python ML service (FastAPI recommended)
2. Add the API URL as an environment variable in Vercel:
\`\`\`
ML_API_URL=https://your-ml-service.com
\`\`\`

## ML Model Integration

The current implementation uses mock predictions for demonstration. To integrate your trained models:

1. Deploy your Python ML service (using the scripts in the original codebase)
2. Update \`app/api/predict/route.ts\` to call your ML service
3. Ensure your ML service returns predictions in the expected format:

\`\`\`json
{
  "prediction": "Toxic" | "Non-Toxic",
  "probability": 0.85,
  "confidence": 0.92,
  "model_used": "Random Forest Ensemble",
  "features_used": 2214,
  "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"
}
\`\`\`

## Project Structure

\`\`\`
├── app/
│   ├── api/
│   │   └── predict/          # Prediction API routes
│   ├── layout.tsx             # Root layout
│   ├── page.tsx               # Home page
│   └── globals.css            # Global styles with theme
├── components/
│   ├── ui/                    # shadcn/ui components
│   ├── prediction-form.tsx    # Single prediction form
│   ├── prediction-result.tsx  # Result display component
│   ├── batch-analysis.tsx     # Batch processing interface
│   └── model-info.tsx         # Model information page
└── lib/
    └── utils.ts               # Utility functions
\`\`\`

## Model Information

### Algorithms
- Random Forest (Primary)
- Gradient Boosting
- Logistic Regression
- Support Vector Machine

### Features
- RDKit molecular descriptors (MW, LogP, TPSA, etc.)
- Morgan fingerprints (2048-bit, radius 2)
- MACCS keys (166-bit)

### Performance Metrics
- Accuracy: ~92%
- ROC-AUC: ~0.94
- Precision: ~89%
- Recall: ~91%

## Data Sources

- **ToxCast/Tox21**: EPA CompTox Chemistry Dashboard
- **ToxRefDB**: In vivo DART toxicity labels
- **RDKit**: Molecular descriptor generation

## Research Use Only

This tool is intended for research and screening purposes only. Predictions should be validated through appropriate experimental methods before making regulatory or clinical decisions. Not approved for diagnostic or therapeutic use.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT License - See LICENSE file for details
\`\`\`
