import React, { useEffect, useState } from "react";
import {
  Typography,
  Box,
  Button,
  TextField,
  Container,
  Paper,
  Snackbar,
  Alert,
  CircularProgress,
  Grid,
} from "@mui/material";
import { useNavigate } from "react-router-dom";
import axios from "axios";

const Profile = () => {
  const [user, setUser] = useState(null);
  const [formData, setFormData] = useState({
    name: "",
    age: "",
    phoneNum: "",
  });
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState("");
  const [snackbarSeverity, setSnackbarSeverity] = useState("success");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchData = async () => {
      const token = localStorage.getItem("token");
      if (!token) {
        navigate("/login");
        return;
      }

      try {
        const profileResponse = await axios.get(
          "https://research-project-iota.vercel.app/api/auth/profile",
          {
            headers: { Authorization: `Bearer ${token}` },
          }
        );
        setUser(profileResponse.data);
        setFormData({
          name: profileResponse.data.name,
          age: profileResponse.data.age,
          phoneNum: profileResponse.data.phoneNum,
        });
      } catch (err) {
        console.error("Error fetching data", err);
        setSnackbarMessage("Failed to fetch data");
        setSnackbarSeverity("error");
        setSnackbarOpen(true);
      }
    };

    fetchData();
  }, [navigate]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const token = localStorage.getItem("token");
    try {
      const res = await axios.put(
        "http://localhost:5000/api/auth/update",
        formData,
        {
          headers: { Authorization: `Bearer ${token}` },
        }
      );
      setSnackbarMessage("Profile updated successfully!");
      setSnackbarSeverity("success");
      setSnackbarOpen(true);
      setUser(res.data.user);
    } catch (err) {
      console.error("Error updating profile", err);
      setSnackbarMessage("Failed to update profile");
      setSnackbarSeverity("error");
      setSnackbarOpen(true);
    }
  };

  const handleSnackbarClose = () => {
    setSnackbarOpen(false);
  };

  return (
    <Container
      maxWidth="lg"
      sx={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        minHeight: "100vh",
        py: 4,
      }}
    >
      <Paper
        elevation={3}
        sx={{
          p: 4,
          width: "100%",
          maxWidth: "600px",
          borderRadius: "12px",
          backgroundColor: "#f5f5f5",
        }}
      >
        <Typography
          variant="h4"
          align="center"
          gutterBottom
          sx={{ fontWeight: "bold", color: "#1976d2" }}
        >
          Profile
        </Typography>
        {user ? (
          <Box component="form" onSubmit={handleSubmit}>
            <TextField
              fullWidth
              label="Name"
              name="name"
              value={formData.name}
              onChange={handleChange}
              margin="normal"
              required
              sx={{ mb: 2 }}
            />
            <TextField
              fullWidth
              label="Age"
              name="age"
              value={formData.age}
              onChange={handleChange}
              margin="normal"
              required
              sx={{ mb: 2 }}
            />
            <TextField
              fullWidth
              label="Phone Number"
              name="phoneNum"
              value={formData.phoneNum}
              onChange={handleChange}
              margin="normal"
              required
              sx={{ mb: 3 }}
            />

            {loading && (
              <Box sx={{ mt: 3, textAlign: "center" }}>
                <CircularProgress />
              </Box>
            )}

            <Box
              sx={{
                mt: 3,
                display: "flex",
                justifyContent: "space-between",
                gap: 2,
              }}
            >
              <Button
                type="submit"
                variant="contained"
                color="primary"
                sx={{ flex: 1, py: 1.5 }}
              >
                Update Profile
              </Button>
              <Button
                variant="contained"
                color="secondary"
                onClick={() => navigate("/")}
                sx={{ flex: 1, py: 1.5 }}
              >
                Back to Home
              </Button>
            </Box>
          </Box>
        ) : (
          <Typography align="center">Loading...</Typography>
        )}
      </Paper>

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={handleSnackbarClose}
        anchorOrigin={{ vertical: "top", horizontal: "center" }}
      >
        <Alert
          onClose={handleSnackbarClose}
          severity={snackbarSeverity}
          sx={{ width: "100%" }}
        >
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </Container>
  );
};

export default Profile;
