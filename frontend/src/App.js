import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import { Container } from "@mui/material";
import Login from "./screens/Login";
import Signup from "./screens/Signup";
import Profile from "./screens/Profile";
import Navbar from "./screens/Navbar"; 
import Home from "./screens/Home";

function App() {
  return (
    <Router>
      <Navbar /> {/* Navbar displayed on all pages */}
      <Container sx={{ mt: 10 }}> {/* Ensure content doesn't overlap with fixed Navbar */}
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/profile" element={<Profile />} />
          <Route path="/" element={<Home />} />
        </Routes>
      </Container>
    </Router>
  );
}

export default App;
