import React from "react";
import { AppBar, Toolbar, Typography, Button } from "@mui/material";
import { useNavigate } from "react-router-dom";

const Navbar = () => {
  const navigate = useNavigate();
  const isLoggedIn = !!localStorage.getItem("token"); // Check if user is logged in

  // Handle logout
  const handleLogout = () => {
    localStorage.removeItem("token"); // Remove token from local storage
    navigate("/login"); // Redirect to login page
  };

  return (
    <AppBar position="fixed" sx={{ width: "100%" }}>
      <Toolbar>
        {/* Logo/Title */}
        <Typography
          variant="h6"
          sx={{ flexGrow: 1, cursor: "pointer" }}
          onClick={() => navigate("/")}
        >
          BioScroller   ✏️
        </Typography>

        {/* Conditional Rendering */}
        {isLoggedIn ? (
          <>

            {/* Spacer to push remaining buttons to the right */}
            <div style={{ flexGrow: 15 }}></div>

            {/* Profile Button */}
            <Button color="inherit" onClick={() => navigate("/profile")}>
              Profile
            </Button>

            {/* Logout Button */}
            <Button color="inherit" onClick={handleLogout}>
              Logout
            </Button>
          </>
        ) : (
          // Login Button
          <Button color="inherit" onClick={() => navigate("/login")}>
            Login
          </Button>
        )}
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;