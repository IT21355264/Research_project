import React, { useState, useEffect } from "react";
import axios from "axios";
import { useDropzone } from "react-dropzone";
import { Button, Card, Typography, CircularProgress } from "@mui/material";
import { makeStyles } from "@mui/styles";
import { useNavigate } from "react-router-dom";

// Custom styles using makeStyles
const useStyles = makeStyles((theme) => ({
  container: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    padding: "20px",
    backgroundColor: "#f8f9fa",
    minHeight: "100vh",
  },
  dropzone: {
    border: "2px dashed #6200ee",
    borderRadius: "10px",
    padding: "20px",
    textAlign: "center",
    cursor: "pointer",
    marginBottom: "20px",
    backgroundColor: "#fff",
    "&:hover": {
      borderColor: "#3700b3",
    },
  },
  imagePreview: {
    maxWidth: "100%",
    maxHeight: "300px",
    borderRadius: "10px",
    marginBottom: "20px",
  },
  button: {
    margin: "8px",
    backgroundColor: "#6200ee",
    color: "#fff",
    "&:hover": {
      backgroundColor: "#3700b3",
    },
  },
  resultCard: {
    marginTop: "20px",
    padding: "15px",
    width: "90%",
    maxWidth: "500px",
    backgroundColor: "#fff",
    borderRadius: "10px",
    boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)",
  },
  sketchImage: {
    width: "100%",
    height: "auto",
    margin: "10px 0",
    borderRadius: "10px",
  },
}));

function Home() {
  const classes = useStyles();
  const navigate = useNavigate();
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [recognizedText, setRecognizedText] = useState("");
  const [loading, setLoading] = useState(false);
  const [sketches, setSketches] = useState([]);
  const [pdfBase64, setPdfBase64] = useState(null);

  // Check if the user is logged in
  useEffect(() => {
    const fetchData = async () => {
      const token = localStorage.getItem("token");
      if (!token) {
        navigate("/login");
        return;
      }

      try {
        await axios.get("https://research-project-iota.vercel.app/api/auth/profile", {
          headers: { Authorization: `Bearer ${token}` },
        });
      } catch (err) {
        localStorage.removeItem("token");
        navigate("/login");
      }
    };

    fetchData();
  }, [navigate]);

  // Handle image drop or selection
  const { getRootProps, getInputProps } = useDropzone({
    accept: "image/*",
    onDrop: (acceptedFiles) => {
      const file = acceptedFiles[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = () => {
          const img = new Image();
          img.src = reader.result;
          img.onload = () => {
            // Resize the image to a maximum width of 500px
            const canvas = document.createElement("canvas");
            const maxWidth = 500;
            const scale = maxWidth / img.width;
            canvas.width = maxWidth;
            canvas.height = img.height * scale;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            setImagePreview(canvas.toDataURL("image/jpeg"));
          };
        };
        reader.readAsDataURL(file);
        setImageFile(file);
        setRecognizedText("");
        setSketches([]);
        setPdfBase64(null);
      }
    },
  });

  // Upload image to the backend
  const uploadImage = async () => {
    if (!imageFile) {
      alert("Please select or drop an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("image", imageFile);

    try {
      setLoading(true);
      const token = localStorage.getItem("token");
      const response = await axios.post(
        "http://192.168.1.2:5002/segment_and_recognize",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
            Authorization: `Bearer ${token}`,
          },
        }
      );

      setRecognizedText(response.data.recognized_text || "No text recognized.");

      const sketchBase64Images = response.data.sketches_base64 || [];
      const sketchUris = sketchBase64Images.map((base64, index) => ({
        uri: `data:image/png;base64,${base64}`,
        key: `sketch_${index}`,
      }));
      setSketches(sketchUris);

      setPdfBase64(response.data.pdf_base64);
    } catch (error) {
      alert("Failed to process the image.");
    } finally {
      setLoading(false);
    }
  };

  // Download the PDF
  const downloadPDF = () => {
    if (!pdfBase64) {
      alert("No PDF available to download.");
      return;
    }

    try {
      const byteCharacters = atob(pdfBase64);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: "application/pdf" });

      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = "recognized_text.pdf";
      link.click();
    } catch (error) {
      alert("Failed to download the PDF.");
    }
  };

  return (
    <div className={classes.container}>
      <Typography variant="h4" gutterBottom>
        Handwritten Text Recognition ✍️
      </Typography>

      <div {...getRootProps()} className={classes.dropzone}>
        <input {...getInputProps()} />
        <Typography variant="body1">
          Drag & drop an image here, or click to select one
        </Typography>
      </div>

      {imagePreview && (
        <img src={imagePreview} alt="Selected" className={classes.imagePreview} />
      )}

      <Button
        variant="contained"
        className={classes.button}
        onClick={uploadImage}
        disabled={loading}
      >
        {loading ? <CircularProgress size={24} color="inherit" /> : "Recognize Text"}
      </Button>

      {recognizedText && (
        <Card className={classes.resultCard}>
          <Typography variant="body1">
            <strong>Recognized Text:</strong> {recognizedText}
          </Typography>
        </Card>
      )}

      {sketches.length > 0 && (
        <Card className={classes.resultCard}>
          <Typography variant="h6" gutterBottom>
            Sketches
          </Typography>
          {sketches.map((sketch) => (
            <img key={sketch.key} src={sketch.uri} alt="Sketch" className={classes.sketchImage} />
          ))}
        </Card>
      )}

      {pdfBase64 && (
        <Button variant="contained" className={classes.button} onClick={downloadPDF}>
          Download PDF
        </Button>
      )}
    </div>
  );
}

export default Home;
