import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import {
  AiFillGithub,
  AiFillFacebook,
  AiFillInstagram,
} from "react-icons/ai";
import { FaLinkedinIn } from "react-icons/fa";

function Footer() {
  return (
    // Container for the footer with fluid width
    <Container fluid className="footer">
      <Row>
        {/* Column for the dev credit */}
        <Col md="4" className="footer-copywright">
          <h3>Designed and Developed by Weather Wizards Team</h3>
        </Col>
        {/* Column for the copyright in4 */}
        <Col md="4" className="footer-copywright">
          <h3>Copyright © {2024} Weather Wizards Team</h3>
        </Col>
        {/* Column for the copyright in4 */}
        <Col md="4" className="footer-copywright">
          <h3>Swinburne University of Technology</h3>
        </Col>
      </Row>
    </Container>
  );
}

export default Footer;
