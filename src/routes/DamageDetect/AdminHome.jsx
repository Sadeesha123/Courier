import { useState } from "react";
import { Link } from "react-router-dom";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faPlus, faStarHalfStroke } from "@fortawesome/free-solid-svg-icons";
import Sidepanel from "../../components/sidepanel";
import bg from '../../images/mainbg1.jpg';

function AdminHome() {
  const [formData, setFormData] = useState({
    senderName: "",
    receiverName: "",
    packageId: ""
  });

  const handleInputChange = (event) => {
    const { name, value } = event.target;
    setFormData((prevState) => ({
      ...prevState,
      [name]: value
    }));
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    
  };
  function handleImageUpload(event) {
    const selectedFiles = event.target.files;
    const fileCount = selectedFiles.length;

    if (fileCount < 1 || fileCount > 5) {
      // Display an error message or handle the validation error accordingly
      console.log("Please select between 1 and 5 images.");
      return;
    }
  }
  return (
    <div className="main-body h-screen w-full bg-slate-100">
      <img src={bg} alt="" srcset="" className="object-cover w-[100%] h-[100%] fixed" />
      <div className="main-body-container w-full flex flex-row absolute">
        <Sidepanel />
        <div className="w-5/6 side-panel p-5 md:ml-[300px] ml-16">
          <div className="common-body p-5 flex flex-col h-full items-center justify-center">
            <div className="form-body w-[80%] flex flex-col p-5 mx-auto items-center justify-center bg-white rounded-lg shadow-md shadow-slate-300">
              <form onSubmit={handleSubmit} className="flex flex-col w-full">
                <label
                  htmlFor="sendername"
                  className="mb-2 font-semibold text-gray-600"
                >
                  Sender Name
                </label>
                <input
                  type="text"
                  id="sendername"
                  name="senderName"
                  required
                  value={formData.senderName}
                  onChange={handleInputChange}
                  className="mb-4 p-2 rounded-lg border border-gray-300"
                />

                <label
                  htmlFor="recievername"
                  className="mb-2 font-semibold text-gray-600"
                >
                  Receiver Name
                </label>
                <input
                  type="text"
                  id="recievername"
                  required
                  name="receiverName"
                  value={formData.receiverName}
                  onChange={handleInputChange}
                  className="mb-4 p-2 rounded-lg border border-gray-300"
                />

                <label
                  htmlFor="pid"
                  className="mb-2 font-semibold text-gray-600"
                >
                  Package ID
                </label>
                <input
                  type="text"
                  id="pid"
                  required
                  name="packageId"
                  value={formData.packageId}
                  onChange={handleInputChange}
                  className="mb-4 p-2 rounded-lg border border-gray-300"
                />

                <label
                  htmlFor="dd"
                  className="mb-2 font-semibold text-gray-600"
                >
                  Driver Details
                </label>
                <input
                  type="text"
                  id="dd"
                  required
                  name="driverDetails"
                  value={formData.driverDetails}
                  onChange={handleInputChange}
                  className="mb-4 p-2 rounded-lg border border-gray-300"
                />

                <label htmlFor="date" className="mb-2 font-semibold text-gray-600">
                  Date
                </label>
                <input type="date" id="date" required name="date" className="mb-4 p-2 rounded-lg border border-gray-300" />


                <label htmlFor="time" className="mb-2 font-semibold text-gray-600">
                  Time
                </label>
                <input type="time" id="time" required name="time" className="mb-4 p-2 rounded-lg border border-gray-300" />

                <label
                  htmlFor="pd"
                  className="mb-2 font-semibold text-gray-600"
                >
                  Package Description
                </label>
                <input
                  type="textarea"
                  id="pd"
                  required
                  name="packageDescription"
                  value={formData.packageDescription}
                  onChange={handleInputChange}
                  className="mb-4 p-2 rounded-lg border border-gray-300 h-24"
                />
                <label
                  htmlFor="images"
                  className="mb-2 font-semibold text-gray-600"
                >
                  Upload Image
                </label>
                <input
                  type="file"
                  id="images"
                  name="images"
                  className="mb-4 py-2 pl-5 file:rounded-lg rounded-lg border border-gray-300"
                  accept="image/*"
                  multiple
                  onChange={handleImageUpload}
                  required
                />
                <div className="flex flex-row space-x-3 w-full">
                  {/* <Link
                    to={{
                      pathname: "/PackageDispatch",
                      state: { formData }
                    }}
                    className="w-full"
                  > */}
                  {/* <a href="/PackageDispatch">
                    <button
                      type="submit"
                      className="py-2 px-4 bg-gradient-to-r from-blue-500 to-blue-700 text-white rounded-lg w-52 ml-auto mt-8"
                    >
                      Generate QR
                    </button>
                    </a> */}
                            <div className="flex items-center justify-between z-10">
                                <button
                                    className="text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline w-full bg-gradient-to-r from-red-500 to-red-700 z-10"
                                    type="submit"
                                >
                                    <a href="/PackageDispatch">Generate QR</a>
                                </button>
                            </div>
                  {/* </Link> */}
                </div>
              </form>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AdminHome;
